#!/usr/bin/env node
/**
 * augengine-mcp CLI — the binary `package.json` has always declared.
 *
 * Usage:
 *   augengine-mcp serve --stdio [--export-root DIR]
 *   augengine-mcp tools
 *
 * `serve --stdio` is what an MCP client launches. Nothing may be written to
 * stdout except JSON-RPC frames — diagnostics go to stderr, or the client's
 * parser breaks.
 */

import { startMcpServer, TOOL_MANIFEST, SERVER_NAME, SERVER_VERSION } from "./server.js";

const HELP = `augengine-mcp — Model Context Protocol server for AugEngine

Usage:
  augengine-mcp serve --stdio [--export-root DIR]   Serve over stdin/stdout
  augengine-mcp tools                               Print the tool manifest
  augengine-mcp --help

Claude Code:
  claude mcp add augengine -- augengine-mcp serve --stdio

Claude Desktop / Cursor (mcpServers entry):
  {
    "mcpServers": {
      "augengine": {
        "command": "augengine-mcp",
        "args": ["serve", "--stdio"]
      }
    }
  }
`;

function main(argv: string[]): void {
  const command = argv[0];

  if (command === undefined || command === "--help" || command === "-h" || command === "help") {
    process.stdout.write(HELP);
    return;
  }

  if (command === "tools") {
    process.stdout.write(`${JSON.stringify(TOOL_MANIFEST, null, 2)}\n`);
    return;
  }

  if (command === "serve") {
    let exportRoot: string | undefined;
    for (let i = 1; i < argv.length; i++) {
      if (argv[i] === "--export-root" && argv[i + 1] !== undefined) {
        exportRoot = argv[++i];
      }
    }
    startMcpServer(exportRoot === undefined ? {} : { exportRoot });
    process.stderr.write(
      `[${SERVER_NAME}] MCP stdio server ${SERVER_VERSION} ready — ${TOOL_MANIFEST.length} tools\n`,
    );
    process.stdin.resume();
    return;
  }

  process.stderr.write(`Unknown command: ${command}\n\n${HELP}`);
  process.exitCode = 1;
}

main(process.argv.slice(2));
