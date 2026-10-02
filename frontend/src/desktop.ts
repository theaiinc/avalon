// Desktop-shell detection and the `window.avalon` bridge.
//
// The pages talk to the Tauri shell only through `window.avalon` (see
// vite-env.d.ts), built here from `invoke`/`listen`.
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { openUrl } from '@tauri-apps/plugin-opener';

export const isTauri = '__TAURI_INTERNALS__' in window;

/** True inside the desktop app, where the page is not served by the
 * dashboard and must call it by absolute URL. */
export const isDesktopShell = isTauri;

/** Tauri rejects with a plain string; callers expect `Error.message`. */
function call<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  return invoke<T>(command, args).catch((error) => {
    throw error instanceof Error ? error : new Error(String(error));
  });
}

function isExternal(href: string): boolean {
  try {
    const url = new URL(href, window.location.href);
    return (url.protocol === 'http:' || url.protocol === 'https:' || url.protocol === 'mailto:')
      && url.origin !== window.location.origin;
  } catch {
    return false;
  }
}

export function installDesktopBridge() {
  if (!isTauri || window.avalon) return;

  window.avalon = {
    getRuntimeConfig: () => call<AvalonRuntimeConfig>('runtime_config'),
    quickTest: (payload) => call('quick_test', { payload }),
    onBackendExit: (callback) => {
      let unlisten: (() => void) | undefined;
      let disposed = false;
      listen<{ code: number | null; signal: string | null }>('avalon:backend-exit', (event) => {
        callback(event.payload);
      }).then((fn) => {
        if (disposed) fn();
        else unlisten = fn;
      });
      return () => {
        disposed = true;
        unlisten?.();
      };
    },
  };

  // Route external links and window.open to the system browser instead of
  // navigating the dashboard away.
  document.addEventListener('click', (event) => {
    const anchor = (event.target as Element | null)?.closest?.('a[href]') as HTMLAnchorElement | null;
    if (!anchor || !isExternal(anchor.href)) return;
    event.preventDefault();
    openUrl(anchor.href).catch(() => {});
  }, true);
  window.open = ((url?: string | URL) => {
    if (url && isExternal(String(url))) openUrl(String(url)).catch(() => {});
    return null;
  }) as typeof window.open;
}
