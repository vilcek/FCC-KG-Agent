#!/usr/bin/env python
"""Interactive terminal chat with the LLM Agent + Knowledge Graph plugin.

Features:
  * Automatically (re)builds the local Kùzu + DuckDB demo databases on demand
  * Loads environment variables from a .env file if present
  * Instantiates `KGAgentPlugin` and wires it into `LLMAgent`
  * Simple async REPL supporting slash commands

Slash Commands:
  /help        Show this help
  /exit        Quit
  /rebuild     Rebuild + reseed both databases (graph + timeseries) and recreate plugin
  /steps       Show the function/tool call trace of the last response
  /system <t>  Replace the system prompt (agent instructions) with <t>
  /reset       Recreate the agent preserving current system prompt (clears conversation state)

Regular input lines are sent to the agent. The function/tool call trace prints names,
arguments, and results (when available) for transparency/debugging.

Environment Variables (required):
  AOAI_ENDPOINT
  AOAI_API_KEY
  AOAI_DEPLOYMENT_NAME

Optional Arguments:
  --rebuild        Force rebuild of databases at startup
  --no-plugin      Start without the KG plugin (raw LLM)
  --system PROMPT  Provide/override initial system prompt

Examples:
  python agent_chat.py --rebuild
  python agent_chat.py --system "You are an expert chemical process assistant." 

"""
from __future__ import annotations
import argparse
import asyncio
import os
import sys
import textwrap
import shutil
import pathlib
from typing import Optional

from dotenv import load_dotenv

# Local imports
from databases import KGBuilder, DATABASES_DIR
from plugins import KGAgentPlugin
from agents import LLMAgent

# -------------- Utility Output Helpers --------------

def color(txt: str, c: str) -> str:
    codes = {
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "reset": "\033[0m",
        "magenta": "\033[35m",
    }
    return f"{codes.get(c,'')}{txt}{codes['reset']}" if sys.stdout.isatty() else txt

HELP_TEXT = """Slash Commands:\n  /help        Show this help\n  /exit        Quit\n  /rebuild     Rebuild + reseed databases and recreate plugin+agent\n  /steps       Show tool/function calls from last response\n  /system <t>  Replace system prompt with <t>\n  /reset       Reset conversation (preserve system prompt)\n  /memclear    Clear in-memory conversation without resetting agent\n  /history [N] Show full (or last N) conversation messages\n  /memstats    Show memory usage (tokens/messages)\n  /clear       Clear the screen\n"""

# -------------- Core REPL Logic --------------

class ChatSession:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.builder: Optional[KGBuilder] = None
        self.plugin: Optional[KGAgentPlugin] = None
        self.agent: Optional[LLMAgent] = None
        self.system_prompt: str = args.system or ("""
            You are an expert in answering questions about processes in a FCC petrochemical plant unit.
            Use the provided tools as needed, but when you need to query the knowledge graph or time-series database, prefer the tools that don't require you to write Cypher or SQL directly.
        """)
        self.last_intermediate: Optional[dict] = None

    # -- Setup / Rebuild --
    def _purge_databases_dir(self):
        """Remove every file/subdirectory under the databases folder.

        This is a stronger clean than the internal KGBuilder(rebuild=True) which
        only deletes the graph path and reinitializes schemas. The user requested
        a full wipe when --rebuild is supplied.
        """
        db_dir = pathlib.Path(DATABASES_DIR)
        if not db_dir.exists():
            return
        for child in db_dir.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception as e:  # Best-effort; continue on errors
                print(color(f"Warning: failed to remove {child}: {e}", "yellow"))

    def build_databases(self, rebuild: bool = False):
        do_rebuild = rebuild or self.args.rebuild
        if do_rebuild:
            print(color("Purging databases directory...", "yellow"))
            self._purge_databases_dir()
        self.builder = KGBuilder(rebuild=do_rebuild)
        kconn, dconn = self.builder.connections
        if not self.args.no_plugin:
            self.plugin = KGAgentPlugin(kconn, dconn)
        else:
            self.plugin = None

    def clear_screen(self):
        # Use ANSI escape sequence to clear screen & move cursor home; fallback to os.system
        try:
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            else:
                # Non-interactive: just print separator
                print("\n" + "-" * 60 + "\n")
        except Exception:
            os.system('cls' if os.name == 'nt' else 'clear')

    def make_agent(self):
        required_env = ["AOAI_ENDPOINT", "AOAI_API_KEY", "AOAI_DEPLOYMENT_NAME"]
        missing = [v for v in required_env if not os.getenv(v)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
        self.agent = LLMAgent(
            llm_endpoint=os.getenv("AOAI_ENDPOINT"),
            llm_api_key=os.getenv("AOAI_API_KEY"),
            llm_deployment_name=os.getenv("AOAI_DEPLOYMENT_NAME"),
            agent_name="kg_agent",
            system_prompt=self.system_prompt,
            plugin=self.plugin,
        )

    def rebuild_everything(self):
        print(color("Rebuilding databases and recreating agent...", "yellow"))
        self.build_databases(rebuild=True)
        self.make_agent()
        print(color("Rebuild complete.", "green"))

    def reset_agent(self):
        print(color("Resetting agent (clearing conversation state)...", "yellow"))
        if self.agent and hasattr(self.agent, "reset_memory"):
            try:
                self.agent.reset_memory()  # type: ignore[attr-defined]
            except Exception:
                # Fallback to full recreation
                self.make_agent()
        else:
            self.make_agent()
        print(color("Agent reset.", "green"))

    def clear_memory(self):
        if self.agent and hasattr(self.agent, "reset_memory"):
            try:
                self.agent.reset_memory()  # type: ignore[attr-defined]
                print(color("Conversation memory cleared.", "green"))
            except Exception as e:
                print(color(f"Failed to clear memory: {e}", "yellow"))
        else:
            print(color("Agent memory not available; use /reset instead.", "yellow"))

    def show_history(self, limit: int | None = None):
        if not self.agent or not hasattr(self.agent, "get_history"):
            print(color("History not available.", "yellow"))
            return
        try:
            history = self.agent.get_history(limit=limit)  # type: ignore[attr-defined]
        except Exception as e:
            print(color(f"Failed to fetch history: {e}", "red"))
            return
        if not history:
            print(color("(history empty)", "yellow"))
            return
        print(color(f"Conversation History (last {limit if limit else 'all'} messages):", "cyan"))
        for i, msg in enumerate(history, 1):
            role = msg.get('role', '?')
            content = msg.get('content', '')
            snippet = content if len(content) < 200 else content[:200] + '...'
            print(color(f" {i:02d} [{role}]:", "magenta"), snippet)

    def show_memstats(self):
        if not self.agent or not hasattr(self.agent, "memory_stats"):
            print(color("Memory stats not available.", "yellow"))
            return
        try:
            stats = self.agent.memory_stats()  # type: ignore[attr-defined]
        except Exception as e:
            print(color(f"Failed to fetch memory stats: {e}", "red"))
            return
        print(color("Memory Stats:", "cyan"))
        for k, v in stats.items():
            print(f"  {k}: {v}")

    # -- Command Handling --
    async def handle_line(self, line: str):
        line = line.strip()
        if not line:
            return
        if line.startswith('/'):
            await self._handle_command(line)
            return
        await self._send_to_agent(line)

    async def _handle_command(self, line: str):
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None
        if cmd == '/help':
            print(HELP_TEXT)
        elif cmd == '/exit':
            raise EOFError
        elif cmd == '/rebuild':
            self.rebuild_everything()
        elif cmd == '/steps':
            if not self.last_intermediate:
                print(color("No previous intermediate steps.", "yellow"))
            else:
                self._print_intermediate(self.last_intermediate)
        elif cmd == '/system':
            if not arg:
                print(color("Usage: /system <new system prompt text>", "yellow"))
            else:
                self.system_prompt = arg
                print(color("System prompt updated. Use /reset to apply immediately.", "green"))
        elif cmd == '/reset':
            self.reset_agent()
        elif cmd == '/memclear':
            self.clear_memory()
        elif cmd == '/clear':
            self.clear_screen()
        elif cmd == '/history':
            limit_int = None
            if arg:
                try:
                    limit_int = int(arg)
                except ValueError:
                    print(color("/history argument must be an integer", "yellow"))
            self.show_history(limit=limit_int)
        elif cmd == '/memstats':
            self.show_memstats()
        else:
            print(color(f"Unknown command: {cmd}. Type /help", "red"))
        # Always add a separating blank line after handling a command (except exit which raises)
        if cmd != '/exit':
            print()

    async def _send_to_agent(self, user_text: str):
        if not self.agent:
            print(color("Agent not initialized.", "red"))
            print()
            return
        try:
            response, intermediate = await self.agent(user_text)
            self.last_intermediate = intermediate
            content = response.get('content', '')
            print(color("Agent:", "magenta"), content)
            print()  # blank line after agent output
        except Exception as e:
            print(color(f"Error invoking agent: {e}", "red"))
            print()

    def _print_intermediate(self, intermediate: dict):
        calls = intermediate.get("function_calls", [])
        if not calls:
            print(color("No function/tool calls recorded.", "yellow"))
            print()
            return
        print(color("Function Calls:", "cyan"))
        for i, c in enumerate(calls, 1):
            name = c.get('name')
            args = c.get('arguments')
            result = c.get('result')
            print(color(f" {i}. {name}", "cyan"))
            if args is not None:
                print(textwrap.indent(color("Args:", "green"), prefix="    "))
                print(textwrap.indent(str(args), prefix="      "))
            if result is not None:
                print(textwrap.indent(color("Result:", "green"), prefix="    "))
                # Trim very long outputs
                r_str = str(result)
                if len(r_str) > 1500:
                    r_str = r_str[:1500] + "... [truncated]"
                print(textwrap.indent(r_str, prefix="      "))
        print()

# -------------- Main Entrypoint --------------

async def async_main(args: argparse.Namespace):
    session = ChatSession(args)
    session.build_databases(rebuild=args.rebuild)
    session.make_agent()
    # Clear screen on startup for clean session view
    session.clear_screen()
    print(color("Interactive agent chat. Type /help for commands.", "cyan"))
    loop = asyncio.get_event_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, lambda: input(color("You: ", "green")))
        except (EOFError, KeyboardInterrupt):
            print()  # newline after Ctrl-D/Ctrl-C
            break
        try:
            await session.handle_line(line)
        except EOFError:
            break
        except Exception as e:
            print(color(f"Unexpected error: {e}", "red"))

    print(color("Goodbye!", "cyan"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with the KG-enabled LLM agent.")
    p.add_argument('--rebuild', action='store_true', help='Rebuild & reseed databases at startup')
    p.add_argument('--no-plugin', action='store_true', help='Start without the KG plugin (plain LLM)')
    p.add_argument('--system', type=str, help='Override initial system prompt')
    return p.parse_args(argv)


def main():
    load_dotenv()  # best-effort; ignore if missing
    args = parse_args(sys.argv[1:])
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == '__main__':
    main()
