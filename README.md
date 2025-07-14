# MCP Whisper Transcription Server

An MCP (Model Context Protocol) server that provides audio transcription capabilities using OpenAI's Whisper API.

## Features

- Transcribe audio files using OpenAI's Whisper API
- Support for multiple audio formats
- Configurable transcription parameters (language, prompt, temperature)
- Multiple output formats (text, JSON, SRT, VTT)

## Installation

```bash
npm install
npm run build
```

## Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### As an MCP Server

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "whisper-transcription": {
      "command": "node",
      "args": ["/path/to/mcp-whisper-transcription/dist/index.js"],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Available Tool

The server provides one tool:

- **transcribe_audio**: Transcribe audio files using OpenAI Whisper API
  - `audioFilePath` (required): Path to the audio file to transcribe
  - `language` (optional): The language of the audio (ISO-639-1 format)
  - `prompt` (optional): Optional prompt to guide the transcription
  - `responseFormat` (optional): Format of the transcription output (json, text, srt, verbose_json, vtt)
  - `temperature` (optional): Sampling temperature (0-1)

## Development

```bash
# Install dependencies
npm install

# Run in development mode
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Lint code
npm run lint

# Type check
npm run typecheck
```

## License

MIT