#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import OpenAI from 'openai';

const WHISPER_TOOL = 'transcribe_audio';

interface TranscribeParams {
  audioFilePath: string;
  language?: string;
  prompt?: string;
  responseFormat?: 'json' | 'text' | 'srt' | 'verbose_json' | 'vtt';
  temperature?: number;
}

class WhisperTranscriptionServer {
  private server: Server;
  private openai: OpenAI | null = null;

  constructor() {
    this.server = new Server(
      {
        name: 'whisper-transcription',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  private setupHandlers(): void {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: WHISPER_TOOL,
          description: 'Transcribe audio files using OpenAI Whisper API',
          inputSchema: {
            type: 'object',
            properties: {
              audioFilePath: {
                type: 'string',
                description: 'Path to the audio file to transcribe',
              },
              language: {
                type: 'string',
                description: 'The language of the audio (ISO-639-1 format)',
              },
              prompt: {
                type: 'string',
                description: 'Optional prompt to guide the transcription',
              },
              responseFormat: {
                type: 'string',
                enum: ['json', 'text', 'srt', 'verbose_json', 'vtt'],
                description: 'Format of the transcription output',
                default: 'text',
              },
              temperature: {
                type: 'number',
                description: 'Sampling temperature (0-1)',
                minimum: 0,
                maximum: 1,
                default: 0,
              },
            },
            required: ['audioFilePath'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      if (request.params.name !== WHISPER_TOOL) {
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown tool: ${request.params.name}`
        );
      }

      if (!process.env.OPENAI_API_KEY) {
        throw new McpError(
          ErrorCode.InvalidRequest,
          'OPENAI_API_KEY environment variable is not set'
        );
      }

      if (!this.openai) {
        this.openai = new OpenAI({
          apiKey: process.env.OPENAI_API_KEY,
        });
      }

      const params = request.params.arguments as unknown as TranscribeParams;

      try {
        const transcription = await this.transcribeAudio(params);
        return {
          content: [
            {
              type: 'text',
              text: transcription,
            },
          ],
        };
      } catch (error) {
        throw new McpError(
          ErrorCode.InternalError,
          `Transcription failed: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    });
  }

  private async transcribeAudio(params: TranscribeParams): Promise<string> {
    const { audioFilePath, language, prompt, responseFormat = 'text', temperature = 0 } = params;

    const { default: fs } = await import('fs');
    
    if (!fs.existsSync(audioFilePath)) {
      throw new Error(`Audio file not found: ${audioFilePath}`);
    }

    const fileStream = fs.createReadStream(audioFilePath);

    const transcription = await this.openai!.audio.transcriptions.create({
      file: fileStream,
      model: 'whisper-1',
      language,
      prompt,
      response_format: responseFormat,
      temperature,
    });

    return typeof transcription === 'string' ? transcription : JSON.stringify(transcription);
  }

  async run(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Whisper Transcription MCP server running on stdio');
  }
}

const server = new WhisperTranscriptionServer();
server.run().catch(console.error);