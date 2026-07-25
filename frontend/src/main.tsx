import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, HashRouter } from 'react-router-dom';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {window.location.protocol === 'file:' ? (
      <HashRouter><App /></HashRouter>
    ) : (
      <BrowserRouter><App /></BrowserRouter>
    )}
  </React.StrictMode>,
);
