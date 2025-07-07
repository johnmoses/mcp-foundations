# Model Context Protocol (MCP) Foundations

## Introduction

MCP uses a client-server architecture where the server exposes tools (functions) that AI clients can call. Communication is via JSON-RPC 2.0 messages: requests, responses, and notifications. The server defines tools with schemas describing input parameters and returns structured results. MCP servers act as bridges connecting AI agents to live data or external APIs.
