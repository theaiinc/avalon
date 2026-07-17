const backendDir = '/Volumes/Data/llama-dash/backend';

const cliExamples = [
  '.venv/bin/python agent_cli.py gpu-list',
  '.venv/bin/python agent_cli.py model-search "qwen 0.5b" --limit 5',
  '.venv/bin/python agent_cli.py model-files <repo-id>',
  '.venv/bin/python agent_cli.py model-download <repo-id> <file.gguf> --wait',
  '.venv/bin/python agent_cli.py api-start --device metal',
  '.venv/bin/python agent_cli.py chat --model <local-model-id> --message "Hello"',
];

const mcpCommand = 'LLAMA_DASH_URL=http://127.0.0.1:8771 .venv/bin/python mcp_server.py';

function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="bg-gray-950 border border-gray-700 rounded-lg p-3 text-xs text-gray-200 overflow-x-auto">
      <code>{children}</code>
    </pre>
  );
}

export default function AgentsPage() {
  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">Agents</h2>
      <p className="text-sm text-gray-500 mb-6">
        Connect other agents to Avalon through the JSON CLI, MCP stdio server, or the local OpenAI-compatible API.
      </p>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <section className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-medium mb-2">MCP Server</h3>
          <p className="text-xs text-gray-500 mb-3">
            Start this stdio server from an MCP client configuration. It exposes tools for GPUs, model search/downloads,
            benchmark tasks, and inference API lifecycle.
          </p>
          <CodeBlock>{`cd ${backendDir}\n${mcpCommand}`}</CodeBlock>
        </section>

        <section className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-medium mb-2">HTTP APIs</h3>
          <div className="space-y-3 text-sm">
            <div>
              <div className="text-gray-400 text-xs mb-1">Dashboard API</div>
              <CodeBlock>http://127.0.0.1:8771/api</CodeBlock>
            </div>
            <div>
              <div className="text-gray-400 text-xs mb-1">OpenAI-compatible gateway</div>
              <CodeBlock>http://127.0.0.1:8787/v1</CodeBlock>
            </div>
          </div>
        </section>

        <section className="bg-gray-800 rounded-lg p-4 border border-gray-700 xl:col-span-2">
          <h3 className="text-sm font-medium mb-2">CLI</h3>
          <p className="text-xs text-gray-500 mb-3">
            The CLI emits JSON, so it is safe for agents and scripts to parse.
          </p>
          <CodeBlock>{`cd ${backendDir}\n${cliExamples.join('\n')}`}</CodeBlock>
        </section>
      </div>
    </div>
  );
}
