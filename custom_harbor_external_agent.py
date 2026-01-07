"""
External Agent for vLLM-served models via local vLLM server

This agent integrates a locally running vLLM server with Harbor framework
by implementing the BaseAgent interface. It works with any model served via vLLM.
"""

import os
import json
from typing import Optional
from openai import AsyncOpenAI

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class VLLMAgent(BaseAgent):
    """
    External agent that uses any model served via local vLLM server
    with OpenAI-compatible API.
    """

    def __init__(
        self,
        logs_dir,
        model_name: Optional[str] = None,
        logger = None,
        api_base: str = "http://localhost:8000/v1",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        **kwargs,
    ):
        """
        Initialize the vLLM agent.

        Args:
            logs_dir: Directory for agent logs (required by BaseAgent)
            model_name: Model name/identifier (can also be passed via kwargs)
            logger: Logger instance (optional)
            api_base: Base URL for the vLLM OpenAI-compatible API (can be passed via kwargs)
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments
        """
        # Handle model_name from either direct parameter or kwargs
        # If passed in both places, prefer the direct parameter
        if model_name is None and 'model_name' in kwargs:
            model_name = kwargs.pop('model_name')

        # Also extract api_base from kwargs if provided there (allows override)
        if 'api_base' in kwargs:
            api_base = kwargs.pop('api_base')

        # Call parent constructor
        super().__init__(logs_dir=logs_dir, model_name=model_name, logger=logger, **kwargs)

        # Set model to use - must be provided
        if not model_name:
            raise ValueError("model_name is required and must match the model served by vLLM")
        self.model = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client pointing to local vLLM server
        self.client = AsyncOpenAI(
            api_key="dummy-key-for-local-server",  # vLLM doesn't require real API key
            base_url=api_base,
        )

    @staticmethod
    def name() -> str:
        """Return the agent's name."""
        return "vllm-agent"

    @staticmethod
    def version() -> Optional[str]:
        """Return the agent's version."""
        return "2.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """
        Setup the agent and its tools before execution.

        Args:
            environment: The execution environment
        """
        # Verify the vLLM server is accessible
        try:
            # Test connection by listing models
            models = await self.client.models.list()
            print(f"Connected to vLLM server at {self.api_base}")
            print(f"Available models: {[model.id for model in models.data]}")
        except Exception as e:
            print(f"Warning: Could not connect to vLLM server at {self.api_base}: {e}")
            print("Please ensure the vLLM server is running before executing tasks.")

    async def _get_formatted_system_prompt(self, environment: BaseEnvironment) -> str:
        """
        Build a dynamic system prompt with environment context.
        Based on DeepAgents approach for providing upfront directory context.

        Args:
            environment: The execution environment

        Returns:
            Formatted system prompt string with current directory and file listing
        """
        # Get current working directory
        pwd_result = await environment.exec("pwd")
        current_directory = pwd_result.stdout.strip() if pwd_result.stdout else "/app"

        # Get file listing
        ls_result = await environment.exec("ls -1")
        files = []
        if ls_result.stdout:
            files = [f for f in ls_result.stdout.strip().split('\n') if f]

        # Build file listing section
        if not files:
            file_listing_header = "Current directory is empty."
            file_listing = ""
        elif len(files) <= 10:
            file_listing_header = f"Files in current directory ({len(files)} files):"
            file_listing = "\n".join(f"{i+1}. {f}" for i, f in enumerate(files))
        else:
            file_listing_header = f"Files in current directory (showing first 10 of {len(files)}):"
            file_listing = "\n".join(f"{i+1}. {f}" for i, f in enumerate(files[:10]))

        # System prompt template (based on DeepAgents approach)
        system_message = """You are an autonomous agent executing tasks in a sandboxed environment. Follow these instructions carefully.

## WORKING DIRECTORY & ENVIRONMENT CONTEXT

Your current working directory is:
{current_directory}

{file_listing_header}
{file_listing}

**IMPORTANT**: This directory information is provided for your convenience at the start of the task. You should:
- Use this information to understand the initial environment state
- Avoid redundantly calling `ls` or similar commands just to list the same directory
- Only use file listing commands if you need updated information (after creating/deleting files) or need to explore subdirectories
- Work in the /app directory unless explicitly instructed otherwise

## Core Guidelines:
1. **Action-Oriented**: Focus on executing commands to complete the task rather than providing explanations
2. **Iterative Problem-Solving**: Break down complex tasks into manageable steps
3. **Command Execution**: Provide bash commands in ```bash code blocks for execution
4. **Result Analysis**: Carefully analyze command outputs to determine next steps
5. **Error Handling**: If a command fails, analyze the error and adapt your approach
6. **Task Completion**: When the task is fully complete, explicitly state "TASK COMPLETE"

## Best Practices:
- Use clear, well-formed bash commands
- Install any required dependencies as needed
- Verify command outputs before proceeding
- Track your progress towards task completion
- Be concise and direct in your responses

## Task Completion:
- Complete the given task following the instructions precisely
- Once you've verified the task is complete, respond with "TASK COMPLETE"
- Do not provide unnecessary explanations unless specifically asked

Remember: Your goal is to complete the task efficiently through terminal commands."""

        return system_message.format(
            current_directory=current_directory,
            file_listing_header=file_listing_header,
            file_listing=file_listing
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Execute the agent with the given instruction using iterative problem-solving.

        Args:
            instruction: The task to complete
            environment: The execution context
            context: Storage for results
        """
        # Build dynamic system prompt with environment context (DeepAgents approach)
        system_prompt = await self._get_formatted_system_prompt(environment)

        # Create response log file in logs directory
        response_log_path = os.path.join(self.logs_dir, "response.txt")
        response_log = open(response_log_path, "w", encoding="utf-8")

        # Log system prompt at the start
        response_log.write("="*80 + "\n")
        response_log.write("SYSTEM PROMPT\n")
        response_log.write("="*80 + "\n\n")
        response_log.write(system_prompt)
        response_log.write("\n\n" + "="*80 + "\n")
        response_log.write("TASK INSTRUCTION\n")
        response_log.write("="*80 + "\n\n")
        response_log.write(instruction)
        response_log.write("\n\n" + "="*80 + "\n")
        response_log.write("AGENT EXECUTION LOG\n")
        response_log.write("="*80 + "\n\n")
        response_log.flush()

        # Unlimited iterations (like Letta) - will run until task completion
        # No max_iterations limit

        # Track conversation history with system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]

        iteration_count = 0
        task_complete = False

        print(f"\n{'='*60}")
        print(f"STARTING TASK: {instruction[:100]}...")
        print(f"{'='*60}\n")

        # Iterative problem-solving loop (unlimited iterations)
        while True:
            iteration_count += 1
            print(f"\n--- Iteration {iteration_count} ---")

            # Log iteration start
            response_log.write(f"\n{'='*60}\n")
            response_log.write(f"ITERATION {iteration_count}\n")
            response_log.write(f"{'='*60}\n\n")
            response_log.flush()

            # Get response from the model
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                assistant_message = response.choices[0].message.content
                print(f"\nAssistant response ({len(assistant_message)} chars):")
                print(f"{assistant_message[:300]}...")
                if len(assistant_message) > 300:
                    print(f"... [{len(assistant_message) - 300} more characters]")

                # Log assistant response
                response_log.write(f"[ASSISTANT RESPONSE]\n")
                response_log.write(assistant_message)
                response_log.write("\n\n")
                response_log.flush()

                # Add assistant response to history
                messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })

                # Check for task completion signal
                if "TASK COMPLETE" in assistant_message.upper():
                    print("\n✓ Agent signaled task completion")
                    task_complete = True
                    break

                # Extract and execute bash commands
                commands = self._extract_bash_commands(assistant_message)

                if not commands:
                    # No commands to execute - ask agent to continue or confirm completion
                    print("\nNo commands found. Asking agent to proceed...")
                    messages.append({
                        "role": "user",
                        "content": "No bash commands detected. Based on the task description, If the task is complete, say 'TASK COMPLETE'. Otherwise, provide the next bash command to execute. Do not stop until the task is fully and correctly completed."
                    })
                    continue

                # Execute each command and collect feedback
                all_outputs = []
                for cmd_idx, cmd in enumerate(commands, 1):
                    print(f"\n[Command {cmd_idx}/{len(commands)}]")
                    print(f"Executing: {cmd[:100]}...")

                    result = await environment.exec(cmd)

                    # Build execution feedback
                    exec_feedback = f"Command {cmd_idx}: {cmd}\n"
                    exec_feedback += f"Exit code: {result.return_code}\n"

                    if result.stdout:
                        stdout_preview = result.stdout[:1000]
                        exec_feedback += f"STDOUT:\n{stdout_preview}"
                        if len(result.stdout) > 1000:
                            exec_feedback += f"\n... [truncated {len(result.stdout) - 1000} chars]"
                        exec_feedback += "\n"
                        print(f"✓ Exit {result.return_code} | stdout: {len(result.stdout)} chars")

                    if result.stderr:
                        stderr_preview = result.stderr[:1000]
                        exec_feedback += f"STDERR:\n{stderr_preview}"
                        if len(result.stderr) > 1000:
                            exec_feedback += f"\n... [truncated {len(result.stderr) - 1000} chars]"
                        exec_feedback += "\n"
                        print(f"✗ Exit {result.return_code} | stderr: {len(result.stderr)} chars")

                    all_outputs.append(exec_feedback)

                # Send all command results back to the model for reflection
                combined_feedback = "\n---\n".join(all_outputs)
                combined_feedback += "\n\nBased on these results, what should you do next? If the task is complete, say 'TASK COMPLETE'."

                messages.append({
                    "role": "user",
                    "content": combined_feedback
                })

            except Exception as e:
                print(f"\n✗ Error in iteration {iteration_count}: {e}")
                # Add error to conversation so agent can adapt
                messages.append({
                    "role": "user",
                    "content": f"An error occurred: {str(e)}\nPlease try a different approach."
                })

        # Final status
        if task_complete:
            status = "completed_with_signal"
        else:
            status = "completed"

        print(f"\n{'='*60}")
        print(f"TASK FINISHED: {status} after {iteration_count} iterations")
        print(f"{'='*60}\n")

        # Log final summary
        response_log.write(f"\n{'='*80}\n")
        response_log.write(f"TASK COMPLETED\n")
        response_log.write(f"{'='*80}\n")
        response_log.write(f"Status: {status}\n")
        response_log.write(f"Total iterations: {iteration_count}\n")
        response_log.write(f"Total messages in conversation: {len(messages)}\n")
        response_log.write(f"Task complete signal detected: {task_complete}\n")
        response_log.write(f"{'='*80}\n")
        response_log.flush()
        response_log.close()

        # Store final result in context
        if context.metadata is None:
            context.metadata = {}
        context.metadata["iterations"] = iteration_count
        context.metadata["conversation_length"] = len(messages)
        context.metadata["status"] = status
        context.metadata["task_complete_signal"] = task_complete

    def _extract_bash_commands(self, text: str) -> list[str]:
        """
        Extract bash commands from markdown code blocks.

        Args:
            text: Text potentially containing ```bash or ```sh code blocks

        Returns:
            List of bash commands to execute
        """
        commands = []
        lines = text.split('\n')
        in_code_block = False
        current_block = []

        for line in lines:
            if line.strip().startswith('```bash') or line.strip().startswith('```sh'):
                in_code_block = True
                current_block = []
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                if current_block:
                    # Join multi-line commands
                    commands.append('\n'.join(current_block))
                    current_block = []
            elif in_code_block:
                current_block.append(line)

        return commands


# Export the agent class
__all__ = ['VLLMAgent']

# Backwards compatibility alias
NvidiaNemotronAgent = VLLMAgent
