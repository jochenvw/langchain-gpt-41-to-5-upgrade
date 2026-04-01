# LangChain GPT 4.1 → GPT 5.1 Upgrade Investigation

## Purpose

Investigate the work required to upgrade a LangChain agent that uses Bring-Your-Own-Data (BYOD) from GPT 4.1 to GPT 5.1, as GPT 4.1 will be deprecated.

## Context

A customer reported that BYOD (On Your Data) breaks when the model is upgraded from GPT 4.1 to GPT 5.1. This repo reproduces the issue and documents the upgrade path.

## Architecture Overview

- **LangChain app** connecting to Azure OpenAI via APIM gateway
- **GPT 4.1** model deployment (target: upgrade to GPT 5.1)
- **Azure AI Search** for BYOD (On Your Data) with index `safety-source-index`
- **Endpoint**: Azure OpenAI through APIM gateway

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file from `.env.example` with the required API keys:
   ```bash
   cp .env.example .env
   ```
   Fill in your Azure OpenAI, APIM gateway, and Azure AI Search credentials.

## Running

_To be documented as the investigation progresses._

## Known Issues

- BYOD (On Your Data) feature breaks when switching from GPT 4.1 to GPT 5.1
- See [UPGRADE_NOTES.md](UPGRADE_NOTES.md) for detailed findings

## Upgrade Notes

See [UPGRADE_NOTES.md](UPGRADE_NOTES.md) for a full breakdown of differences, reproduction steps, and resolution.
