# JavaScript/TypeScript Style Guide

## Code Style

### Formatter and Linter

Use **ESLint** for linting and **Prettier** for formatting:

```bash
npm run lint        # Lint with auto-fix
npm run format      # Format code
npm run typecheck   # TypeScript type checking
```

Run all quality checks:

```bash
npm run check       # Runs lint, format, and typecheck in parallel
```

### Line Length

100 characters (configured in prettier.config.js)

### Quotes

Double quotes for strings, single quotes for JSX attributes:

**Good:**
```typescript
const name = "EDM Analyzer";
const message = "Analysis complete";

// JSX
<Button label='Click Me' onClick={handler} />
```

**Bad:**
```typescript
const name = 'EDM Analyzer';  // Use double quotes
const message = `Analysis complete`;  // No need for template literals here

// JSX
<Button label="Click Me" />  // Use single quotes in JSX
```

### Semicolons

Always use semicolons:

**Good:**
```typescript
const x = 5;
return result;
```

**Bad:**
```typescript
const x = 5
return result
```

### Import Organization

Standard library → Third-party → Internal, sorted alphabetically:

**Good:**
```typescript
import fs from "node:fs";
import path from "node:path";

import express from "express";
import { z } from "zod";

import { analyzeAudio } from "@/lib/audio";
import { logger } from "@/lib/logger";
import type { BPMResult } from "@/types";
```

**Bad:**
```typescript
import { logger } from "@/lib/logger";
import express from "express";
import { analyzeAudio } from "@/lib/audio";
import fs from "node:fs";
import type { BPMResult } from "@/types";
import { z } from "zod";
import path from "node:path";
```

## TypeScript

### Type Annotations

Always use explicit types for function parameters and return types:

**Good:**
```typescript
interface AudioFile {
  path: string;
  duration: number;
  bpm?: number;
}

function analyzeFile(file: AudioFile): Promise<BPMResult> {
  return detectBPM(file.path);
}

// Use type inference for simple local variables
const files = loadAudioFiles();
const count = files.length;
```

**Bad:**
```typescript
// Missing interface
function analyzeFile(file: any): Promise<any> {
  return detectBPM(file.path);
}

// Over-annotation
const files: AudioFile[] = loadAudioFiles();
const count: number = files.length;
```

### Type vs Interface

Use `interface` for object shapes, `type` for unions, intersections, and primitives:

**Good:**
```typescript
// Interfaces for objects
interface User {
  id: string;
  name: string;
}

interface AdminUser extends User {
  permissions: string[];
}

// Types for unions/intersections
type Result<T> = { success: true; data: T } | { success: false; error: string };
type ID = string | number;
type Point = { x: number } & { y: number };
```

**Bad:**
```typescript
// Don't use type for simple objects
type User = {
  id: string;
  name: string;
};

// Don't use interface for unions
interface Result {
  success: boolean;
  data?: any;
  error?: string;
}
```

### Null and Undefined

Use `null` for intentional absence, `undefined` for optional/missing values:

**Good:**
```typescript
interface Config {
  apiKey: string;
  timeout?: number;  // Optional, can be undefined
  cache: CacheConfig | null;  // Intentionally absent
}

function findUser(id: string): User | undefined {
  return users.find(u => u.id === id);
}
```

**Bad:**
```typescript
interface Config {
  apiKey: string;
  timeout: number | null;  // Use undefined for optional
  cache: CacheConfig | undefined;  // Use null for intentional absence
}

function findUser(id: string): User | null {
  return users.find(u => u.id === id) ?? null;
}
```

### Type Assertions

Avoid type assertions (`as`) except when necessary. Prefer type guards:

**Good:**
```typescript
function isAudioFile(file: unknown): file is AudioFile {
  return (
    typeof file === "object" &&
    file !== null &&
    "path" in file &&
    typeof file.path === "string"
  );
}

if (isAudioFile(data)) {
  // TypeScript knows data is AudioFile here
  analyzeFile(data);
}
```

**Bad:**
```typescript
const file = data as AudioFile;  // Unsafe
analyzeFile(file);
```

### Enums

Prefer const objects or string literal unions over enums:

**Good:**
```typescript
// String literal union
type Status = "pending" | "processing" | "complete" | "error";

// Const object for values
const Status = {
  Pending: "pending",
  Processing: "processing",
  Complete: "complete",
  Error: "error",
} as const;

type StatusValue = typeof Status[keyof typeof Status];
```

**Bad:**
```typescript
enum Status {
  Pending = "pending",
  Processing = "processing",
  Complete = "complete",
  Error = "error",
}
```

## Modern JavaScript

### Use Modern Syntax

Prefer modern ES2015+ features:

**Good:**
```typescript
// Destructuring
const { name, age } = user;
const [first, ...rest] = items;

// Spread operator
const merged = { ...defaults, ...options };
const combined = [...arr1, ...arr2];

// Template literals
const message = `User ${name} is ${age} years old`;

// Arrow functions
const double = (x: number) => x * 2;
files.map(f => analyzeFile(f));

// Optional chaining
const bpm = result?.audio?.bpm;

// Nullish coalescing
const timeout = config.timeout ?? 3000;
```

**Bad:**
```typescript
// Old syntax
const name = user.name;
const age = user.age;

var merged = Object.assign({}, defaults, options);
var combined = arr1.concat(arr2);

var message = "User " + name + " is " + age + " years old";

var double = function(x) { return x * 2; };

var bpm = result && result.audio && result.audio.bpm;

var timeout = config.timeout !== null && config.timeout !== undefined
  ? config.timeout
  : 3000;
```

### Async/Await

Prefer `async/await` over raw promises:

**Good:**
```typescript
async function analyzeFiles(paths: string[]): Promise<BPMResult[]> {
  const results: BPMResult[] = [];

  for (const path of paths) {
    try {
      const result = await analyzeFile(path);
      results.push(result);
    } catch (error) {
      logger.error(`Failed to analyze ${path}`, { error });
    }
  }

  return results;
}

// Parallel processing
async function analyzeParallel(paths: string[]): Promise<BPMResult[]> {
  return Promise.all(paths.map(p => analyzeFile(p)));
}
```

**Bad:**
```typescript
function analyzeFiles(paths: string[]): Promise<BPMResult[]> {
  return Promise.resolve([])
    .then(results => {
      return paths.reduce((promise, path) => {
        return promise.then(acc => {
          return analyzeFile(path)
            .then(result => [...acc, result])
            .catch(error => {
              logger.error(`Failed: ${path}`, error);
              return acc;
            });
        });
      }, Promise.resolve(results));
    });
}
```

## Error Handling

### Custom Errors

Create typed error classes:

**Good:**
```typescript
class AppError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode: number = 500
  ) {
    super(message);
    this.name = this.constructor.name;
    Error.captureStackTrace(this, this.constructor);
  }
}

class AudioLoadError extends AppError {
  constructor(path: string, cause?: Error) {
    super(`Failed to load audio file: ${path}`, "AUDIO_LOAD_ERROR", 400);
    this.cause = cause;
  }
}

// Usage
try {
  const audio = await loadAudio(path);
} catch (error) {
  if (error instanceof AudioLoadError) {
    logger.error("Audio load failed", { error });
  }
  throw error;
}
```

**Bad:**
```typescript
// Generic errors
throw new Error("Failed to load file");

// Loose error handling
try {
  const audio = await loadAudio(path);
} catch (error: any) {
  console.log(error.message);
}
```

### Error Handling Patterns

Always handle errors explicitly:

**Good:**
```typescript
// Result type pattern
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

async function safeAnalyze(path: string): Promise<Result<BPMResult>> {
  try {
    const value = await analyzeFile(path);
    return { ok: true, value };
  } catch (error) {
    return { ok: false, error: error as Error };
  }
}

// Using the result
const result = await safeAnalyze(path);
if (result.ok) {
  console.log(`BPM: ${result.value.bpm}`);
} else {
  logger.error("Analysis failed", { error: result.error });
}
```

**Bad:**
```typescript
async function safeAnalyze(path: string): Promise<BPMResult | null> {
  try {
    return await analyzeFile(path);
  } catch {
    return null;  // Lost error information
  }
}
```

## Data Validation

### Use Zod for Runtime Validation

Prefer Zod for schema validation:

**Good:**
```typescript
import { z } from "zod";

const AudioFileSchema = z.object({
  path: z.string().min(1),
  duration: z.number().positive(),
  bpm: z.number().min(60).max(200).optional(),
  confidence: z.number().min(0).max(1).optional(),
});

type AudioFile = z.infer<typeof AudioFileSchema>;

function parseAudioFile(data: unknown): AudioFile {
  return AudioFileSchema.parse(data);
}

// Safe parsing
const result = AudioFileSchema.safeParse(data);
if (result.success) {
  const file: AudioFile = result.data;
} else {
  logger.error("Validation failed", { errors: result.error.errors });
}
```

**Bad:**
```typescript
interface AudioFile {
  path: string;
  duration: number;
  bpm?: number;
}

function parseAudioFile(data: any): AudioFile {
  // No runtime validation
  return data as AudioFile;
}
```

## React (if applicable)

### Functional Components

Use functional components with hooks:

**Good:**
```typescript
interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

function Button({ label, onClick, disabled = false }: ButtonProps) {
  return (
    <button onClick={onClick} disabled={disabled}>
      {label}
    </button>
  );
}

// With hooks
function AudioPlayer({ file }: { file: AudioFile }) {
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);

  useEffect(() => {
    // Setup and cleanup
    const interval = setInterval(() => {
      if (playing) {
        setPosition(p => p + 0.1);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [playing]);

  return (
    <div>
      <button onClick={() => setPlaying(!playing)}>
        {playing ? "Pause" : "Play"}
      </button>
      <progress value={position} max={file.duration} />
    </div>
  );
}
```

**Bad:**
```typescript
// Class components (deprecated)
class Button extends React.Component<ButtonProps> {
  render() {
    return (
      <button onClick={this.props.onClick}>
        {this.props.label}
      </button>
    );
  }
}
```

### Hooks Rules

Follow hooks rules:

**Good:**
```typescript
function Component() {
  // Always call hooks at top level
  const [state, setState] = useState(0);
  const value = useMemo(() => expensiveCalc(state), [state]);

  useEffect(() => {
    // Effect logic
  }, [state]);

  return <div>{value}</div>;
}
```

**Bad:**
```typescript
function Component() {
  if (condition) {
    // Don't call hooks conditionally
    const [state, setState] = useState(0);
  }

  return <div>...</div>;
}
```

## Documentation

### JSDoc Comments

Use JSDoc for public APIs:

**Good:**
```typescript
/**
 * Analyzes an audio file to detect BPM.
 *
 * @param path - Path to the audio file
 * @param options - Analysis options
 * @returns Promise resolving to BPM result
 * @throws {AudioLoadError} If the file cannot be loaded
 * @throws {BPMDetectionError} If BPM detection fails
 *
 * @example
 * ```typescript
 * const result = await analyzeBPM("track.mp3", { minBpm: 60 });
 * console.log(`Detected BPM: ${result.bpm}`);
 * ```
 */
export async function analyzeBPM(
  path: string,
  options?: AnalysisOptions
): Promise<BPMResult> {
  // Implementation
}
```

**Bad:**
```typescript
// No documentation
export async function analyzeBPM(path: string, options?: AnalysisOptions) {
  // Implementation
}

// Over-documented (types already clear)
/**
 * @param path string - Path to file
 * @param options AnalysisOptions | undefined - Options object or undefined
 * @returns Promise<BPMResult> - A promise that resolves to BPMResult
 */
export async function analyzeBPM(path: string, options?: AnalysisOptions) {
  // Implementation
}
```

## Testing

### Use Vitest

Prefer Vitest for modern testing:

**Good:**
```typescript
import { describe, it, expect, beforeEach, vi } from "vitest";

describe("BPM Detector", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should detect BPM correctly", async () => {
    const result = await analyzeBPM("test.mp3");
    expect(result.bpm).toBeCloseTo(128, 1);
    expect(result.confidence).toBeGreaterThan(0.8);
  });

  it("should handle missing files", async () => {
    await expect(analyzeBPM("missing.mp3"))
      .rejects.toThrow(AudioLoadError);
  });

  it("should respect min/max BPM options", async () => {
    const result = await analyzeBPM("test.mp3", {
      minBpm: 120,
      maxBpm: 140,
    });
    expect(result.bpm).toBeGreaterThanOrEqual(120);
    expect(result.bpm).toBeLessThanOrEqual(140);
  });
});
```

**Bad:**
```typescript
// Jest (outdated)
test("BPM detection", () => {
  const result = analyzeBPM("test.mp3");
  expect(result.bpm).toBe(128);  // No async handling
});

// Poor test structure
it("test", async () => {
  const r = await analyzeBPM("test.mp3");
  expect(r).toBeTruthy();  // Not specific
});
```

## Node.js Specific

### File System Operations

Use `node:fs/promises` for async file operations:

**Good:**
```typescript
import { readFile, writeFile } from "node:fs/promises";

async function loadConfig(path: string): Promise<Config> {
  const data = await readFile(path, "utf-8");
  return JSON.parse(data);
}

async function saveResults(path: string, results: BPMResult[]): Promise<void> {
  await writeFile(path, JSON.stringify(results, null, 2));
}
```

**Bad:**
```typescript
import fs from "fs";

function loadConfig(path: string): Config {
  const data = fs.readFileSync(path, "utf-8");  // Blocking
  return JSON.parse(data);
}
```

### Path Operations

Use `node:path` for path manipulation:

**Good:**
```typescript
import { join, resolve, basename, extname } from "node:path";

const audioPath = resolve(join(baseDir, "audio", filename));
const name = basename(audioPath, extname(audioPath));
```

**Bad:**
```typescript
const audioPath = baseDir + "/audio/" + filename;  // Wrong on Windows
const name = audioPath.split("/").pop()?.replace(/\.[^.]+$/, "");
```

## Logging

Use structured logging (pino, winston):

**Good:**
```typescript
import pino from "pino";

const logger = pino({
  level: process.env.LOG_LEVEL || "info",
});

logger.info({ file: path, bpm: result.bpm }, "BPM analysis complete");
logger.error({ error, file: path }, "Failed to analyze file");
```

**Bad:**
```typescript
console.log(`Analyzed ${path}: ${result.bpm} BPM`);
console.error("Error:", error);  // No context
```

## Package Management

Use modern package managers (pnpm preferred):

```bash
pnpm install           # Install dependencies
pnpm add <package>     # Add dependency
pnpm add -D <package>  # Add dev dependency
pnpm run <script>      # Run script
```

## Summary

Key points:
- TypeScript with strict mode enabled
- ESLint + Prettier for code quality
- Prefer modern ES2015+ syntax
- Explicit types for APIs, inference for locals
- Zod for runtime validation
- Async/await over promises
- Structured logging
- Vitest for testing
- pnpm for package management
