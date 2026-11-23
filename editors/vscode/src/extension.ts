// expex shapes for VS Code: on save of a .py file, run
//     python -m expex shapes <file> --json
// and show its hints as inlay hints at the end of the lines, and its failures
// as diagnostics. The output format is described in src/expex/cli/shapes.py.

import { execFile, ChildProcess } from "child_process";
import * as path from "path";
import * as vscode from "vscode";

interface Hint {
  file: string;
  line: number; // 1-based
  column: number; // 0-based
  end_column: number;
  text: string;
}

interface Diagnostic {
  file: string;
  line: number;
  column: number;
  end_column: number;
  severity: "error" | "warning" | "information";
  message: string;
}

interface Result {
  hints: Hint[];
  diagnostics: Diagnostic[];
}

const SEVERITY: Record<string, vscode.DiagnosticSeverity> = {
  error: vscode.DiagnosticSeverity.Error,
  warning: vscode.DiagnosticSeverity.Warning,
  information: vscode.DiagnosticSeverity.Information,
};

const hints = new Map<string, Hint[]>(); // document uri -> its hints
const running = new Map<string, ChildProcess>();
const hintsChanged = new vscode.EventEmitter<void>();
let diagnostics: vscode.DiagnosticCollection;
let log: vscode.OutputChannel;

function config(): vscode.WorkspaceConfiguration {
  return vscode.workspace.getConfiguration("expex");
}

/** The interpreter the Python extension selected for this file, else `expex.python`, else `python`. */
async function interpreter(uri: vscode.Uri): Promise<string> {
  const ext = vscode.extensions.getExtension("ms-python.python");
  if (ext) {
    try {
      const api: any = ext.isActive ? ext.exports : await ext.activate();
      const envs = api?.environments;
      if (envs?.getActiveEnvironmentPath) {
        const active = envs.getActiveEnvironmentPath(uri);
        const env = await envs.resolveEnvironment(active);
        const exe: string | undefined = env?.executable?.uri?.fsPath ?? active?.path;
        if (exe) {
          return exe;
        }
      }
      const details = api?.settings?.getExecutionDetails?.(uri);
      if (details?.execCommand?.length) {
        return details.execCommand[0];
      }
    } catch (e) {
      log.appendLine(`could not ask the Python extension for an interpreter: ${e}`);
    }
  }
  return config().get<string>("python") || "python";
}

async function check(doc: vscode.TextDocument): Promise<void> {
  if (doc.languageId !== "python" || doc.uri.scheme !== "file") {
    return;
  }
  const key = doc.uri.toString();
  running.get(key)?.kill(); // a newer save wins
  const python = await interpreter(doc.uri);
  const file = doc.uri.fsPath;
  const cwd = vscode.workspace.getWorkspaceFolder(doc.uri)?.uri.fsPath ?? path.dirname(file);
  const args = ["-m", "expex", "shapes", file, "--json"];
  log.appendLine(`${python} ${args.join(" ")}`);
  const started = Date.now();
  const child = execFile(
    python,
    args,
    { cwd, maxBuffer: 64 * 1024 * 1024, timeout: config().get<number>("timeout", 120) * 1000 },
    (err, stdout, stderr) => {
      if (running.get(key) !== child) {
        return; // superseded
      }
      running.delete(key);
      let result: Result;
      try {
        result = JSON.parse(stdout);
      } catch {
        // exit code 1 with JSON on stdout just means "there are errors"; no JSON is a real failure
        log.appendLine(`expex failed${err ? ` (${err.message.split("\n")[0]})` : ""}:\n${stderr}`);
        if (/No module named expex/.test(stderr)) {
          vscode.window.showWarningMessage(
            `expex is not installed for ${python}. Select another interpreter, or set expex.python.`
          );
        }
        return;
      }
      log.appendLine(
        `${path.basename(file)}: ${result.hints.length} hints, ${result.diagnostics.length} diagnostics ` +
          `in ${((Date.now() - started) / 1000).toFixed(2)} s`
      );
      show(doc.uri, result);
    }
  );
  running.set(key, child);
}

function show(uri: vscode.Uri, result: Result): void {
  hints.set(uri.toString(), result.hints);
  diagnostics.set(
    uri,
    result.diagnostics.map((d) => {
      const range = new vscode.Range(d.line - 1, d.column, d.line - 1, d.end_column);
      const diag = new vscode.Diagnostic(range, d.message, SEVERITY[d.severity] ?? vscode.DiagnosticSeverity.Error);
      diag.source = "expex";
      return diag;
    })
  );
  hintsChanged.fire();
}

/** One hint per line, at its end: `f32[B, T]`, or `h: f32[D], c: f32[D]` when a line assigns several names. */
class ShapeHints implements vscode.InlayHintsProvider {
  onDidChangeInlayHints = hintsChanged.event;

  provideInlayHints(doc: vscode.TextDocument, range: vscode.Range): vscode.InlayHint[] {
    if (!config().get<boolean>("hints.enabled", true)) {
      return [];
    }
    const byLine = new Map<number, Hint[]>();
    for (const h of hints.get(doc.uri.toString()) ?? []) {
      const line = h.line - 1;
      if (line < range.start.line || line > range.end.line || line >= doc.lineCount) {
        continue;
      }
      byLine.set(line, [...(byLine.get(line) ?? []), h]);
    }
    const out: vscode.InlayHint[] = [];
    for (const [line, hs] of byLine) {
      const text = doc.lineAt(line).text;
      const label =
        hs.length === 1
          ? hs[0].text
          : hs
              .map((h) => (h.text.startsWith("->") ? h.text : `${text.slice(h.column, h.end_column)}: ${h.text}`))
              .join(", ");
      const hint = new vscode.InlayHint(new vscode.Position(line, text.length), label, vscode.InlayHintKind.Type);
      hint.paddingLeft = true;
      out.push(hint);
    }
    return out;
  }
}

export function activate(context: vscode.ExtensionContext): void {
  log = vscode.window.createOutputChannel("expex");
  diagnostics = vscode.languages.createDiagnosticCollection("expex");
  context.subscriptions.push(
    log,
    diagnostics,
    hintsChanged,
    vscode.languages.registerInlayHintsProvider({ language: "python", scheme: "file" }, new ShapeHints()),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (config().get<boolean>("runOnSave", true)) {
        check(doc);
      }
    }),
    vscode.workspace.onDidChangeTextDocument((e) => {
      // hints are by line number: once lines move, they are stale until the next save
      if (e.contentChanges.some((c) => c.text.includes("\n") || c.range.start.line !== c.range.end.line)) {
        if (hints.delete(e.document.uri.toString())) {
          hintsChanged.fire();
        }
      }
    }),
    vscode.workspace.onDidCloseTextDocument((doc) => {
      hints.delete(doc.uri.toString());
      diagnostics.delete(doc.uri);
    }),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("expex.hints")) {
        hintsChanged.fire();
      }
    }),
    vscode.commands.registerCommand("expex.check", () => {
      const doc = vscode.window.activeTextEditor?.document;
      if (doc) {
        check(doc);
      }
    }),
    vscode.commands.registerCommand("expex.toggleHints", async () => {
      const on = config().get<boolean>("hints.enabled", true);
      await config().update("hints.enabled", !on, vscode.ConfigurationTarget.Global);
    })
  );
}

export function deactivate(): void {
  for (const child of running.values()) {
    child.kill();
  }
}
