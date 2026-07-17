/// <reference types="vite/client" />

interface AvalonRuntimeConfig {
  dashboardUrl: string;
  gatewayUrl: string;
  publicGatewayUrl: string;
  apiKey: string;
  packaged: boolean;
}

interface Window {
  avalon?: {
    getRuntimeConfig: () => Promise<AvalonRuntimeConfig>;
    quickTest: (payload: {
      model: string;
      messages: { role: string; content: string }[];
      max_tokens: number;
      format: 'openai' | 'anthropic';
    }) => Promise<any>;
    onBackendExit: (callback: (payload: { code: number | null; signal: string | null }) => void) => () => void;
  };
}
