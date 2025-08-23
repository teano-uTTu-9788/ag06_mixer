import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import { rateLimit } from 'express-rate-limit';
import { createPrometheusMiddleware } from '@bmatei/prometheus-middleware';
import { trace, context, SpanStatusCode } from '@opentelemetry/api';
import { registerInstrumentations } from '@opentelemetry/instrumentation';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';
import pino from 'pino';
import pinoHttp from 'pino-http';
import { z } from 'zod';
import { PrismaClient } from '@prisma/client';
import Redis from 'ioredis';
import { config } from './config';
import { errorHandler } from './middleware/errorHandler';
import { authMiddleware } from './middleware/auth';
import { mixerRouter } from './routes/mixer';
import { presetsRouter } from './routes/presets';
import { analyticsRouter } from './routes/analytics';
import { healthRouter } from './routes/health';
import { MixerService } from './services/MixerService';
import { AudioEngine } from './services/AudioEngine';
import { MidiController } from './services/MidiController';
import { WebSocketHandler } from './handlers/WebSocketHandler';
import { MetricsCollector } from './services/MetricsCollector';
import { CacheManager } from './services/CacheManager';

// Initialize logger
const logger = pino({
  level: config.logLevel,
  transport: {
    target: 'pino-pretty',
    options: {
      colorize: true,
      ignore: 'pid,hostname',
      translateTime: 'SYS:standard',
    },
  },
});

// Initialize OpenTelemetry
const initTelemetry = () => {
  const jaegerExporter = new JaegerExporter({
    endpoint: config.jaegerEndpoint,
  });

  const resource = new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: 'ag06-mixer-backend',
    [SemanticResourceAttributes.SERVICE_VERSION]: config.version,
  });

  const sdk = new NodeSDK({
    resource,
    spanProcessor: new BatchSpanProcessor(jaegerExporter),
  });

  sdk.start();
  logger.info('OpenTelemetry initialized');
};

// Initialize Prisma
const prisma = new PrismaClient({
  log: ['error', 'warn'],
  errorFormat: 'minimal',
});

// Initialize Redis
const redis = new Redis(config.redisUrl, {
  retryStrategy: (times) => Math.min(times * 50, 2000),
  reconnectOnError: (err) => {
    const targetError = 'READONLY';
    if (err.message.includes(targetError)) {
      return true;
    }
    return false;
  },
});

redis.on('error', (err) => {
  logger.error({ err }, 'Redis error');
});

redis.on('connect', () => {
  logger.info('Redis connected');
});

// Initialize services
const cacheManager = new CacheManager(redis);
const metricsCollector = new MetricsCollector();
const audioEngine = new AudioEngine(logger);
const midiController = new MidiController(logger);
const mixerService = new MixerService(
  prisma,
  cacheManager,
  audioEngine,
  midiController,
  logger
);

// Create Express app
const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: config.corsOrigins,
    credentials: true,
  },
  transports: ['websocket', 'polling'],
});

// Middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", 'data:', 'https:'],
    },
  },
}));

app.use(compression());
app.use(cors({
  origin: config.corsOrigins,
  credentials: true,
}));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging
app.use(pinoHttp({
  logger,
  autoLogging: {
    ignore: (req) => req.url === '/health',
  },
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP',
  standardHeaders: true,
  legacyHeaders: false,
});

app.use('/api', limiter);

// Prometheus metrics
const prometheusMiddleware = createPrometheusMiddleware({
  app,
  collectDefaultMetrics: true,
  collectGCMetrics: true,
  requestDurationBuckets: [0.1, 0.5, 1, 1.5, 2, 3, 5, 10],
  customLabels: {
    service: 'ag06-mixer',
    version: config.version,
  },
});

app.use(prometheusMiddleware);

// Health check (before auth)
app.use('/health', healthRouter);

// Authentication middleware for protected routes
app.use('/api', authMiddleware);

// API Routes
app.use('/api/mixer', mixerRouter(mixerService));
app.use('/api/presets', presetsRouter(mixerService));
app.use('/api/analytics', analyticsRouter(metricsCollector));

// WebSocket handling
const wsHandler = new WebSocketHandler(io, mixerService, logger);
wsHandler.initialize();

// Error handling
app.use(errorHandler(logger));

// Graceful shutdown
const gracefulShutdown = async (signal: string) => {
  logger.info({ signal }, 'Shutting down gracefully');
  
  // Stop accepting new connections
  server.close(() => {
    logger.info('HTTP server closed');
  });
  
  // Close WebSocket connections
  io.close(() => {
    logger.info('WebSocket server closed');
  });
  
  // Close database connections
  await prisma.$disconnect();
  logger.info('Database disconnected');
  
  // Close Redis connection
  redis.disconnect();
  logger.info('Redis disconnected');
  
  // Close audio/MIDI connections
  audioEngine.shutdown();
  midiController.shutdown();
  
  process.exit(0);
};

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Unhandled rejection handling
process.on('unhandledRejection', (reason, promise) => {
  logger.error({ reason, promise }, 'Unhandled Rejection');
  // Don't exit in production, but alert monitoring
  metricsCollector.incrementCounter('unhandled_rejections');
});

process.on('uncaughtException', (error) => {
  logger.fatal({ error }, 'Uncaught Exception');
  // Exit after logging
  process.exit(1);
});

// Start server
const startServer = async () => {
  try {
    // Initialize telemetry
    if (config.telemetryEnabled) {
      initTelemetry();
    }
    
    // Test database connection
    await prisma.$connect();
    logger.info('Database connected');
    
    // Initialize audio engine
    await audioEngine.initialize();
    logger.info('Audio engine initialized');
    
    // Initialize MIDI controller
    await midiController.initialize();
    logger.info('MIDI controller initialized');
    
    // Start server
    app.listen(config.port, () => {
      logger.info(
        {
          port: config.port,
          env: config.env,
          version: config.version,
        },
        '🚀 AG06 Mixer Backend Server started'
      );
    });
  } catch (error) {
    logger.fatal({ error }, 'Failed to start server');
    process.exit(1);
  }
};

// Start the server
startServer();