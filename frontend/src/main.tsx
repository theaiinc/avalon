import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, HashRouter } from 'react-router-dom';
import App from './App';
import { installDesktopBridge, isDesktopShell } from './desktop';
import './index.css';

installDesktopBridge();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {isDesktopShell ? (
      <HashRouter><App /></HashRouter>
    ) : (
      <BrowserRouter><App /></BrowserRouter>
    )}
  </React.StrictMode>,
);
