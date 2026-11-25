# `.chat.md` File Format

This is the authoritative specification of the `.chat.md` conversation file
format. It is written for both human readers and LLM agents that create or edit
these files.

**Canonical example:**

```markdown
---
model: local-model
endpoint: http://localhost:9000/v1/chat/completions
---
**System**

You are terse.

---
**User** | 2026-11-22 14:30

Hello!

---
**Assistant** | 2026-11-22 14:31 | model: local-model

Hi there.
```

A file consists of an **optional YAML frontmatter block** followed by a body
of **sections**. Every section is one role (System, User, Assistant, or Tool)
plus its content. The parser is strict: every violation is rejected with an
error that names the offending 1-based line number (see
[§4 Error catalog](#4-complete-catalog-of-syntax--validation-errors)).

---

## 1. Overall file structure

```
file      := [frontmatter] section ('---'-intro section)*
section   := header '\n' content
header    := '**' role '**' [metadata]        (see §2)
role      := 'System' | 'User' | 'Assistant' | 'Tool'   (case-insensitive)
```

### 1.1 Frontmatter

A file has frontmatter **if and only if its first line is a fence**: exactly
`---` or `...`, ignoring trailing whitespace (a leading space disqualifies the
line). The block ends at the next fence line (`---` or `...`, same rules).
Everything between the fences is YAML.

- An **empty block** (`---` immediately closed by `---`) is valid and parses
  to all-default fields.
- A file whose first line is a fence but that **never closes** the block is an
  error (the opening fence's line number is reported).
- **Unknown YAML keys are ignored** on parse, so extra fields do not break
  old or foreign files.

Supported fields (in serialization order):

| Field         | Type   | Default  | Notes                                                                 |
| ------------- | ------ | -------- | --------------------------------------------------------------------- |
| `model`       | string | *(none)* | Model name to request.                                                |
| `endpoint`    | string | *(none)* | Full `/v1/chat/completions` URL.                                      |
| `tools`       | bool   | `true`   | Enable built-in tools. **Lenient deserialization**: any non-boolean YAML value (e.g. the legacy free-form `tools:` list) reads as `true`. |
| `scope`       | string | `folder` | `folder` or `vault`: which directory file tools operate in.           |
| `network`     | bool   | `false`  | Allow `fetch_url` and shared-net sandboxed bash.                      |
| `allow_shell` | bool   | `false`  | Allow the `bash` tool.                                                |

Serialization rules: the block is written only if the source file had one;
fields are emitted in the table's order; a field is written **only when it
differs from its default** (so `tools: true`, `scope: folder`,
`network: false`, `allow_shell: false` never appear); an all-default block
serializes as an empty `---\n---\n`.

### 1.2 The body and its sections

- The **first line of the body** (the line after the frontmatter's closing
  fence, or the file's first line when there is no frontmatter) must be a
  header for a `**System**` or `**User**` section — nothing else. In
  particular, no preamble text, no `**Assistant**`, and no `**Tool**` may lead
  the file.
- The first section carries **no leading horizontal rule of its own**: the
  frontmatter's closing fence (or the start of the file) serves that visual
  purpose.
- **Every later section** is introduced by a `---` horizontal-rule line
  immediately followed by the section's header line. The rule must be
  preceded by a blank (or whitespace-only) line; otherwise it is a parse
  error (§4.2). In canonical output the separator is exactly `\n\n---\n`,
  even when the preceding section's content is empty.
- A `**System**` section is **only valid as the first section**. It is
  forbidden mid-conversation because every request must carry exactly one
  system message, always first: the hierarchical `AGENTS.md` context followed
  by this section's content. A second system message elsewhere would be
  rejected or repositioned by model chat templates.
- A section's **content** is everything between its header line and the next
  section's `---` rule (or end of file), trimmed of surrounding whitespace.
  Interior lines are preserved verbatim. Content may be empty.

### 1.3 Roles

| Role        | Meaning                                                        |
| ----------- | -------------------------------------------------------------- |
| `System`    | Per-file system instructions; first section only; folded into the single leading system message, never sent as conversation content. |
| `User`      | A user prompt.                                                 |
| `Assistant` | A model reply (reasoning in a collapsible block, §3.4; issued tool calls in trailing blocks, §3.3). |
| `Tool`      | A persisted tool result (§3.3).                                |

Role names in headers are matched case-insensitively and tolerate surrounding
whitespace (`** USER **` parses as `User`).

---

## 2. Header & metadata syntax

Every section starts with a header line:

```
**<Role>** [ | <timestamp> ] [ | <key>: <value> ]*
```

- The role is bold: `**User**`, `**Assistant**`, `**System**`, or `**Tool**`.
- Metadata is **optional**; when present it consists of plain-text segments
  separated by **exactly** ` | ` (space, pipe, space). No other separator is
  accepted.
- The **first** metadata segment may be a bare timestamp with no key prefix.
- **Every** later segment must be a `key: value` pair.
- No bold or italic markers anywhere in the metadata; no `|` inside a
  segment.

### 2.1 Timestamps

A timestamp is exactly 16 characters of the form `YYYY-MM-DD HH:MM`: ASCII
digits everywhere except `-` at positions 5 and 8, a space at position 11, and
`:` at position 14 (1-based). Examples: `2026-11-23 22:42`.

Rules:

- At most **one** timestamp per header.
- It must be the **first** metadata segment (immediately after the role).
- It is optional; headers may carry only `key: value` pairs.

### 2.2 Metadata tags

Each non-timestamp segment is `key: value`:

- The key is the text before the **first** colon, trimmed; it must be
  non-empty.
- The value is everything after that first colon, trimmed — so values may
  themselves contain colons (`model: unsloth/Qwen:Q8` → key `model`, value
  `unsloth/Qwen:Q8`).
- An empty value is allowed (`flag:` or `flag: ` parses as key `flag` with an
  empty value).
- A key may appear **at most once** per header; a repeated key is a
  header-format error (§4.3). Keys are case-sensitive (`model` and `Model`
  are different keys).
- Tag order is preserved. Well-known tags: `model` (which model produced the
  turn), `thinking` (assistant-only switch for the collapsible reasoning
  block — see below and §3.4), `tool_calls` (assistant-only count of the
  trailing Tool Call blocks — see below and §3.3), and `tool` (the tool
  name) / `id` (the tool call id), which are **`**Tool**`-only** and required
  there (§3.3.2).

The `thinking` tag is special:

- It is **only allowed on `**Assistant**` headers**; on any other role it is a
  header-format error (§4.3).
- Its value must be **exactly the lowercase literal `true`**; anything else
  (`false`, `True`, empty, …) is a header-format error (§4.3).
- When present, the turn's body must carry the collapsible reasoning block of
  §3.4; when absent, the whole body is verbatim content (no reasoning).

The `tool_calls` tag is special:

- It is **only allowed on `**Assistant**` headers**; on any other role it is a
  header-format error (§4.3).
- Its value must be a **positive integer**; `0`, negatives, decimals, and
  non-integers (`true`, `1.5`, empty, …) are header-format errors (§4.3).
- When present with value `N ≥ 1`, the turn's body must end with exactly `N`
  well-formed Tool Call blocks (§3.3); otherwise a line-numbered error (§4.2).
- The tag is emitted only when the turn carries at least one tool call; the
  canonical tag order is `model`, then `thinking`, then `tool_calls`.

### 2.3 Whitespace handling

- Leading whitespace before the opening `**` is ignored; trailing whitespace
  at end of line is ignored.
- The separator must be exactly one space on each side of the pipe: `**User**|
  …`, `**User** |…`, `**User**  | …` (double space), and `**User**· …` are all
  errors.
- A header may be just `**Role**` with no metadata at all.

### 2.4 Valid vs. invalid headers

```markdown
**User**                                              ← valid (no metadata)
**System**                                            ← valid (first section only)
**User** | 2026-11-23 22:42                           ← valid (timestamp only)
**User** | model: m1                                  ← valid (tags, no timestamp)
**Assistant** | 2026-11-23 22:42 | model: local-model ← valid (canonical form)
**Assistant** | model: m1 | thinking: true | tool_calls: 2 ← valid
**Tool** | 2026-11-23 22:42 | tool: bash | id: call_1 ← valid

**User** · *2026-11-23 22:42*                         ← INVALID: '·' separator
**User** - 2026-11-23 22:42                           ← INVALID: '-' separator
**User**| 2026-11-23 22:42                            ← INVALID: missing spaces
**User** | *2026-11-23 22:42*                         ← INVALID: italicized segment
**User** | 2026-11-23 22:42 | note                    ← INVALID: no 'key: ' prefix
**User** | model: m | 2026-11-23 22:42                ← INVALID: timestamp not first
**User** | 2026-11-23 22:42 | 2026-11-23 22:43        ← INVALID: duplicate timestamp
**User** | : value                                    ← INVALID: empty key
```

---

## 3. Message bodies, fenced code blocks & tool blocks

### 3.1 Fenced code blocks protect `---` lines

Section boundaries are only recognized **outside** fenced code blocks. The
fence tracker implements a CommonMark subset:

- An opening fence is a line (indented by at most 3 spaces) starting with at
  least three backticks `` ` `` or tildes `~`. A backtick fence's info string
  may not contain backticks; a tilde fence's may.
- The fence stays open until a **closing** fence: same character, a run at
  least as long as the opener, and nothing but whitespace after it.
- Lines indented by more than 3 spaces are never fence delimiters.
- A line whose first non-space character is a backslash is never a fence
  delimiter; such lines are escape artifacts of §3.2 and parse as content.
- While inside a fence, every line — including `---` lines and lines that look
  like headers — is plain content.

Consequently, a message body may quote or contain example conversation
fragments without them being mistaken for section boundaries:

````markdown
**User**

Here is an example:

```markdown
---
**Assistant** | 2026-11-22 14:31

this is just text in a fence
```
````

### 3.2 Boundary detection and lossless escaping

A **boundary** is a `---` horizontal-rule line (exactly `---`, surrounding
whitespace tolerated) that is outside any fence, immediately followed by a
line that *begins* with `**`, and preceded by a blank or whitespace-only
line. The following line must start with the two asterisks at column 0 — an
indented role-looking line does not form a boundary. Three consequences:

- A `---` line whose next line does not begin with `**` is ordinary content (a
  Markdown horizontal rule inside the message), whether or not a blank line
  precedes it.
- A `---` on the very last line of a section has no following line, so it is
  content too.
- A `---` directly above a line that begins with `**` but **not** preceded by
  a blank line is a format error (§4.2). The serializer never emits this:
  every separator it writes is `\n\n---\n`, so there is always a blank line
  above the rule — even when the preceding section's content is empty.

Because of this, message content that itself contains such a pattern would be
misread on the next parse. The serializer therefore **escapes** colliding
lines with a bijective odd-backslash rule, and the parser inverts it exactly:

- **Turn-boundary collisions.** An unfenced line consisting of `N >= 0`
  leading backslashes followed by `---` (and optional trailing whitespace),
  directly before a line that begins with `**`, is rewritten on serialize
  with `2 * N + 1` leading backslashes: `0 -> 1` (`\---`), `1 -> 3`
  (`\\\---`), `2 -> 5` (`\\\\\---`). On parse, such a line must carry an
  **odd** count `k >= 1` of leading backslashes; the parser restores
  `(k - 1) / 2` backslashes (`1 -> 0`, `3 -> 1`, `5 -> 2`) and treats the
  line as content. An even count is a format error (§4.2). Because escaped
  lines are content, they are exempt from the blank-line rule above: user
  content such as `Some text\n---\n**Bold**` (no blank line) serializes as
  `Some text\n\---\n**Bold**` and parses back byte-for-byte.
- **Code-fence collisions.** An unfenced line starting with `N >= 1`
  backslashes immediately followed by three or more `` ` `` or `~` characters
  is rewritten on serialize with `2 * N + 1` leading backslashes (the same
  mapping). On parse, an odd count is restored as above; an even count is a
  format error (§4.2).
- **Unclosed opening fences.** If an opening fence reaches the end of a
  message body without a matching closing fence, the serializer prepends one
  backslash to the opener (```` ```rust ```` becomes ```` \```rust ````) and
  appends **no** synthetic closing fence. On parse the backslashed opener is
  not a fence delimiter: the odd-backslash rule above applies (`1 -> 0`), so
  the single added backslash is stripped and the opener — and everything
  after it — comes back as plain content exactly as written. Unclosed fences
  are therefore 100% lossless, byte-for-byte.

Lines inside a fenced code block are never inspected or escaped. The rules
are bijective: for every message body `c`, parsing the serialized form
restores `c` byte-for-byte (`parse(to_markdown(conv)) == conv`).

### 3.3 Tool calls and tool turns

A model's tool calls are persisted **on the issuing `**Assistant**` turn**;
each executed result is a separate `**Tool**` section (§3.3.2).

#### 3.3.1 Tool Call blocks (assistant turns)

An assistant turn that issued tool calls carries the header tag
`tool_calls: <N>` (§2.2) and ends its body with exactly `N` well-formed
**Tool Call blocks**, one per call, in the order the model issued them.
Canonical form — reasoning block first (when present), then the visible
answer, then the blocks; adjacent pieces separated by a single blank line:

```markdown
**Assistant** | 2026-11-23 22:42 | model: local-model | thinking: true | tool_calls: 2

<details>
<summary>Thinking</summary>

{reasoning}

</details>

{answer}

<details>
<summary>Tool Call: <code>read_file</code> (<code>call_1</code>)</summary>

~~~~json
{"path": "a.md"}
~~~~

</details>

<details>
<summary>Tool Call: <code>bash</code> (<code>call_2</code>)</summary>

~~~~json
{"command": "ls"}
~~~~

</details>
```

- The summary line is exactly
  `<summary>Tool Call: <code>{name}</code> (<code>{id}</code>)</summary>`;
  `name` and `id` are non-empty and contain no whitespace.
- The arguments sit in a **tilde fence** whose info string is `json` (or
  empty), exactly as the model sent them. The serializer picks a fence run of
  `max(4, longest embedded tilde run + 1)`, so embedded tildes cannot
  terminate it early; the parser accepts any opening run of at least three
  tildes and closes on a bare tilde run at least as long.
- The blocks are emitted **verbatim** (no boundary escaping): their fences
  are self-contained, so a `---` inside the arguments is protected from
  section-boundary detection by the fence (§3.1).

**Parsing** (only for `**Assistant**` sections carrying `tool_calls: N`, and
only after the reasoning block of §3.4 has been stripped): extraction walks
**backward from the end of the section**, peeling off one well-formed block
per iteration, `N` times:

- The last non-blank line must be a bare `</details>`; otherwise a
  line-numbered error (§4.2).
- Above it (skipping blank lines) comes the closing tilde fence — a bare run
  of at least three tildes.
- Walking further up, the nearest line that parses as a Tool Call summary is
  the block's summary; below the summary, the first tilde fence (info string
  `json` or empty) opens the arguments and must not be longer than the
  closing fence.
- The first non-blank line above the summary must be a bare `<details>`.
- The arguments are the lines between the two fences, joined and trimmed.

Each extracted block is removed from the region before the next iteration, so
**literal examples of Tool Call blocks inside the visible content are left
untouched** — only the trailing `N` blocks count (extra well-formed blocks
beyond `N` remain part of the content). If fewer than `N` well-formed blocks
can be extracted, parsing fails with a line-numbered error (§4.2). The
remaining lines, trimmed, are the turn's visible content. Without the
`tool_calls` tag, no details parsing happens at all: `<details>` blocks in an
assistant body are verbatim content.

#### 3.3.2 Tool turns (`**Tool**`)

A persisted tool result is a `**Tool**` section whose header carries the
required `tool:` and `id:` tags — the call's name and id, which live **only**
in the header, never in the body — and whose body has this exact shape:

```markdown
**Tool** | 2026-11-23 22:42 | tool: bash | id: call_42

<details>
<summary>Output</summary>

~~~~text
exit code: 0
a b
~~~~

</details>
```

- The output sits in a **tilde fence** whose info string is `text` (or
  empty), with the same fence sizing rule as Tool Call blocks.
- Validation is **strict**: the header must carry non-empty `tool:` and
  `id:` tags, and the body must be exactly this wrapper — a bare `<details>`
  line immediately followed by a bare `<summary>Output</summary>` line, then
  the tilde fence (a run of at least three tildes), and finally a bare
  `</details>` line; blank lines may separate the pieces. Any deviation is a
  line-numbered parse error (§4.2, §4.3).

**Wire-format pairing** (`to_messages`): each tool turn becomes a
`role: "tool"` message carrying `tool_call_id` (the `id:` tag) and the
unwrapped output; the call itself (id, name, arguments) comes from the
**issuing assistant turn's** parsed Tool Call blocks. Since `parse()`
rejects malformed tool turns, every parsed tool turn projects cleanly; only
hand-constructed turns can fail the well-formedness predicate, and those are
skipped defensively.

`to_messages` is a **pure projection**: it never synthesizes messages. When
an assistant turn carries more tool calls than there are following `**Tool**`
turns answering them (e.g. execution was cancelled mid-way), the dangling
calls simply stay unanswered in the projected list. The request-prep layer
(`crate::llm::complete_tool_calls`, run by the CLI and TUI before every
request) closes the gap: for every `tool_call_id` without a matching tool
message it appends a synthetic `role: "tool"` message with content
`Tool execution cancelled or interrupted`.

### 3.4 Reasoning blocks (`thinking: true`)

Assistant turns from thinking models persist their reasoning in a collapsible
`<details>` block. The `**Assistant**` header must carry the tag
`thinking: true` (exactly lowercase, §2.2); **without the tag the whole body
is verbatim content**, even if it looks exactly like a wrapper.

Canonical form (reasoning followed by a visible answer):

```markdown
**Assistant** | 2026-11-23 22:42 | model: local-model | thinking: true

<details>
<summary>Thinking</summary>

{reasoning}

</details>

{answer}
```

When the answer is empty, the trailing blank line and the answer are omitted —
the body ends at the `</details>` line.

**Serialization.** The wrapper is emitted only when the reasoning is non-empty
after trimming. Each piece (reasoning, answer) is escaped **independently**
for boundary collisions (§3.2), so fence state never leaks from one piece into
the other. Stray details tags inside the reasoning are additionally escaped so
a later parse extracts exactly this text:

- Tag matching is **case-insensitive** and happens only **outside** fenced
  code blocks and inline backtick spans (a run of N backticks opens a span
  closed by the next run of exactly N; an unclosed run extends to end of
  line). An *opener* is `<details` — no whitespace between `<` and `details` —
  followed by `>` or by attributes before `>`; a *closer* is `</details`
  followed only by optional whitespace before `>`. `< details>` and
  `</ details>` are not tags.
- A tag already preceded by `k >= 1` backslashes is rewritten with `2 * k + 1`
  backslashes (the same bijective rule as §3.2) and has **no effect on the
  nesting stack**.
- An unescaped opener is remembered; an unescaped closer pops the stack when a
  matching opener exists (a matched nested pair is left untouched) or gets one
  backslash prepended when it does not. Openers still on the stack at the end
  of the reasoning also get one backslash prepended.

**Parsing** (only for `**Assistant**` sections carrying `thinking: true`):

- The **first non-blank line** of the body must trim to exactly `<details>`
  (lowercase), immediately followed by a line that trims to exactly
  `<summary>Thinking</summary>`; otherwise a line-numbered error (§4.4).
- A forward scan from there tracks fenced code blocks and inline backtick
  spans, starting at depth 1: an unescaped opener increments the depth, an
  unescaped closer decrements it. The wrapper ends at the first closer that
  brings the depth to zero, and that line must trim to exactly `</details>`
  (lowercase) — a closer sharing its line with other text, or carrying
  attributes/whitespace before `>`, is an error (§4.4).
- Backslashes immediately before a details tag inside the reasoning must come
  in an **odd** count `k >= 1`; `(k - 1) / 2` are restored (the inverse of the
  serializer's rewrite). An even count is an error (§4.4).
- The lines between the summary line and the closing line (boundary escapes
  restored, §3.2) are the **reasoning**; the lines after the closing line
  (trimmed, escapes restored) are the **visible content**.

---

## 4. Complete catalog of syntax & validation errors

`parse()` fails fast: it stops at the first violation and reports it through a
single helper that renders `line {N}: {message}`, where **N is the 1-based
physical line number in the file** (frontmatter lines count; the first body
line is the closing fence's line + 1). Callers add the file path, so CLI
output looks like:

```
Error: failed to parse notes.chat.md

Caused by:
    line 5: invalid header metadata separator; expected ' | '
```

### 4.1 Frontmatter errors

| Error message | Condition | Example (invalid) |
| ------------- | --------- | ----------------- |
| `line 1: frontmatter block is not closed with a `---` line` | The first line is a fence (`---` or `...`) but no later line closes the block. Always reported at line 1 (the opening fence). | `---\nmodel: m\nno closing fence` |
| `invalid YAML frontmatter (lines 2..=N)` (plus the underlying YAML error) | The block is closed, but the YAML is malformed or a known field has the wrong type. N is the closing fence's line. Unknown keys are **not** an error. | `---\nmodel: [unclosed\n---\n**User**\n\nhi` |

### 4.2 Body-structure errors

| Error message | Condition | Example (invalid) |
| ------------- | --------- | ----------------- |
| `line N: file has no sections; expected a **System** or **User** header` | The body is empty — an empty file, or frontmatter with nothing after it. | `` (empty file) |
| `line N: first section must be a **System** or **User** header` | The first body line is not a role header at all (plain text, an unknown bold role, …), **or** it is a valid header for a role other than System/User (`**Assistant**`, `**Tool**`). | `Just some notes.` / `**Assistant**\n\nhi` |
| `line N: **System** is only allowed as the first section` | A `---` boundary introduces a `**System**` header in any later section. N points at the `**System**` line (one below the rule). | `**User**\n\nhi\n\n---\n**System**\n\nlate` |
| `line N: turn separator '---' must be preceded by a blank line` | An unfenced `---` line directly above a line that begins with `**` whose preceding line is not blank (or whitespace-only). Escaped lines (odd leading backslashes, §3.2) are content and exempt. N points at the `---` line. | `**User**\n\nhi\n---\n**Assistant**\n\nhello` |
| `line N: even number of backslashes before '---'; expected an odd count` | An unfenced line with an even count `k >= 2` of leading backslashes followed by `---`, directly before a line that begins with `**`. Odd counts are valid escapes (§3.2) that restore `(k - 1) / 2` backslashes. | `**User**\n\n\\---\n**Assistant**` |
| `line N: even number of backslashes before a code fence; expected an odd count` | An unfenced line with an even count `k >= 2` of leading backslashes immediately followed by three or more `` ` `` or `~`. Odd counts are valid escapes (§3.2). | `**User**\n\n\\~~~x` |
| `line N: expected <n> tool call blocks at the end of the section; found fewer` | An assistant section with `tool_calls: <n>` runs out of body lines before `<n>` trailing Tool Call blocks are extracted (§3.3.1). N points at the section's last line. | `**User**\n\nq\n\n---\n**Assistant** \| tool_calls: 1\n\n` (empty body) |
| `line N: expected a closing '</details>' line to end the tool call block` | While extracting a trailing Tool Call block, the region's last non-blank line is not a bare `</details>`. | `**Assistant** \| tool_calls: 1\n\n<details>\n<summary>Tool Call: <code>a</code> (<code>c1</code>)</summary>\n\n~~~~json\n{}\n~~~~` (no closer) |
| `line N: malformed tool call block; expected a tilde-fenced arguments block` | The extracted block has no valid fence pair: no closing fence above the closer, no opening fence below the summary, or the opening fence is longer than the closing one. | `**Assistant** \| tool_calls: 1\n\n<details>\n<summary>Tool Call: <code>a</code> (<code>c1</code>)</summary>\n\nnot a fence\n\n</details>` |
| `line N: malformed tool call block; expected a Tool Call summary line` | No line above the closing fence parses as `<summary>Tool Call: <code>{name}</code> (<code>{id}</code>)</summary>` with non-empty, whitespace-free `name` and `id`. | `**Assistant** \| tool_calls: 1\n\n<details>\n\n~~~~json\n{}\n~~~~\n\n</details>` (no summary) |
| `line N: malformed tool call block; expected an opening '<details>' line` | The first non-blank line above the summary is not a bare `<details>`. | `**Assistant** \| tool_calls: 1\n\n<summary>Tool Call: <code>a</code> (<code>c1</code>)</summary>\n\n~~~~json\n{}\n~~~~\n\n</details>` (no opener) |
| `line N: expected a bare '<details>' line to open the tool output block` | A `**Tool**` body's first non-blank line is not a bare `<details>` (or the body is empty). N points at the section's last line. | `**User**\n\nq\n\n---\n**Tool** \| tool: bash \| id: c1\n\n` (empty body) |
| `line N: expected a bare '<summary>Output</summary>' line after the opening tag` | The line after the opening `<details>` does not trim to exactly `<summary>Output</summary>`. | `**Tool** \| tool: bash \| id: c1\n\n<details>\n\n~~~~text\nout\n~~~~\n\n</details>` (no summary) |
| `line N: expected a tilde-fenced output block in the tool turn body` | The first non-blank line below the summary is not a tilde fence (info string `text` or empty), or no fence exists before the section ends. | `**Tool** \| tool: bash \| id: c1\n\n<details>\n<summary>Output</summary>\n\nnot a fence\n\n</details>` |
| `line N: unclosed tool output block; expected a closing tilde fence` | No bare tilde run at least as long as the opener follows it. N points at the section's last line. | `**Tool** \| tool: bash \| id: c1\n\n<details>\n<summary>Output</summary>\n\n~~~~text\nout` (no closing fence) |
| `line N: expected a bare '</details>' line to close the tool output block` | The section's last non-blank line is not a bare `</details>`, or non-blank content follows the closing fence. | `**Tool** \| tool: bash \| id: c1\n\n<details>\n<summary>Output</summary>\n\n~~~~text\nout\n~~~~\nextra\n</details>` (trailing content) |

### 4.3 Header-format errors

All of these apply to a line in *header position* — the first body line, or
the line immediately after an unfenced `---` rule. A line that merely looks
like a header but is not in header position (no preceding rule) is ordinary
content and never triggers them.

| Error message | Condition | Example (invalid) |
| ------------- | --------- | ----------------- |
| `line N: invalid header metadata separator; expected ' \| '` | Content follows the bold role but does not begin with exactly ` | ` — e.g. the legacy `·` separator, `-`, a tab, or missing/doubled spaces around the pipe. **Or** a metadata segment contains a stray `\|` (such as a dangling trailing pipe). | `**User** · *2026-11-23 22:42*` / `**User**| 2026-11-23 22:42` / `**User** \| 2026-11-23 22:42 \|` |
| `line N: header metadata must be plain text; bold/italic markers are not allowed` | Any metadata segment contains a `*`. | `**User** \| *2026-11-23 22:42*` / `**User** \| 2026-11-23 22:42 \| *model: m*` |
| `line N: duplicate timestamp in header` | Two segments both match the timestamp shape. | `**User** \| 2026-11-23 22:42 \| 2026-11-23 22:43` |
| `line N: timestamp must be the first metadata segment after the role` | A timestamp-shaped segment appears after a `key: value` segment. | `**User** \| model: m \| 2026-11-23 22:42` |
| `line N: invalid header metadata; 'key: value' requires a non-empty key` | A segment has a colon but the text before it is empty (after trimming). | `**User** \| : value` |
| `line N: invalid header metadata; expected a timestamp or 'key: value'` | A segment is neither timestamp-shaped nor contains a colon — a bare word without a key prefix. | `**User** \| 2026-11-23 22:42 \| note` |
| `line N: invalid header; expected a bold role like **User**` | Defensive: a line in header position lacks the opening or closing `**`. (Unreachable through `parse()`, which pre-checks the role prefix, but part of the strict parser's contract.) | — |
| `line N: unknown role in header: <text>` | Defensive: the bolded text is not one of System/User/Assistant/Tool. (Same reachability note.) | — |
| `line N: 'thinking' metadata is only allowed on **Assistant** headers` | A `thinking:` tag appears on a `**System**`, `**User**`, or `**Tool**` header (§2.2). | `**User** \| thinking: true` |
| `line N: 'thinking' metadata must be the literal value 'true'` | The `thinking` tag's value is not exactly the lowercase literal `true` (`false`, `True`, empty, …). | `**Assistant** \| thinking: false` |
| `line N: 'tool_calls' metadata is only allowed on **Assistant** headers` | A `tool_calls:` tag appears on a `**System**`, `**User**`, or `**Tool**` header (§2.2). | `**User** \| tool_calls: 1` |
| `line N: 'tool_calls' metadata must be a positive integer` | The `tool_calls` tag's value is not a positive integer (`0`, `-1`, `1.5`, `true`, empty, …). | `**Assistant** \| tool_calls: 0` |
| `line N: duplicate metadata key '<key>' in header` | The same `key:` appears more than once in one header (keys are case-sensitive; timestamps are not keys and have their own duplicate check). | `**Assistant** \| model: a \| model: b` |
| `line N: 'tool' metadata is only allowed on **Tool** headers` | A `tool:` tag appears on a `**System**`, `**User**`, or `**Assistant**` header (§2.2). | `**User** \| tool: bash` |
| `line N: 'id' metadata is only allowed on **Tool** headers` | An `id:` tag appears on a `**System**`, `**User**`, or `**Assistant**` header (§2.2). | `**Assistant** \| id: call_1` |
| `line N: **Tool** header requires a non-empty 'tool:' metadata tag` | A `**Tool**` header lacks a `tool:` tag, or its value is empty (§3.3.2). | `**Tool** \| id: call_1` / `**Tool** \| tool: \| id: call_1` |
| `line N: **Tool** header requires a non-empty 'id:' metadata tag` | A `**Tool**` header lacks an `id:` tag, or its value is empty (§3.3.2). | `**Tool** \| tool: bash` |

### 4.4 Reasoning-block errors

All of these apply to a `**Assistant**` section whose header carries
`thinking: true` (§3.4). N is the offending physical line.

| Error message | Condition | Example (invalid) |
| ------------- | --------- | ----------------- |
| `line N: expected a bare '<details>' line to open the reasoning block` | The first non-blank body line does not trim to exactly `<details>` (lowercase — e.g. `<DETAILS>`). | `**Assistant** \| thinking: true\n\n<DETAILS>\n<summary>Thinking</summary>\n…` |
| `line N: expected a bare '<summary>Thinking</summary>' line after the opening tag` | The line after the opening tag does not trim to exactly `<summary>Thinking</summary>` (wrong case or extra text). | `**Assistant** \| thinking: true\n\n<details>\n<summary>thinking</summary>\n…` |
| `line N: a closing '</details>' must stand alone on its own line` | A closer that brings the nesting depth to zero shares its line with other text, or carries attributes/whitespace before `>`. | `**Assistant** \| thinking: true\n\n<details>\n<summary>Thinking</summary>\n\nr\n\n</details >` |
| `line N: even number of backslashes before a <details> tag; expected an odd count` | An even count `k >= 2` of backslashes immediately before a details tag inside the reasoning. Odd counts restore `(k - 1) / 2` (§3.4). | `**Assistant** \| thinking: true\n\n<details>\n<summary>Thinking</summary>\n\n\\</details>` |
| `line N: unclosed reasoning block; expected a closing '</details>' line` | No closer brings the depth to zero before the end of the section. N points at the section's last line. | `**Assistant** \| thinking: true\n\n<details>\n<summary>Thinking</summary>\n\nreasoning…` |

### 4.5 What is *not* an error

- **Unknown frontmatter keys** — ignored on parse.
- **A `---` whose next line does not begin with `**`** — plain content (a Markdown horizontal rule inside the message).
- **A `---` as the last line of a section** — no following line, so content.
- **A blank-line-preceded `---` directly above a bold line that is not a valid role header** (e.g. `**Bold**`) — not a boundary; both lines are content.
- **A `---` followed by an *indented* role-looking line** — not a boundary (the following line must begin with `**` at column 0); both lines are content.
- **Role-looking lines without a preceding rule** (and not the first body line) — plain content.
- **Escaped collision lines** — odd-backslash `\---`-style lines and backslashed fence lines are content; the parser restores the original backslashes (§3.2).
- **An assistant body that looks like a Thinking wrapper but whose header lacks `thinking: true`** — the whole body is verbatim content; no reasoning is extracted and no §4.4 error fires.
- **`<details>` blocks in an assistant body whose header lacks `tool_calls:`** — verbatim content; no Tool Call extraction happens and no §3.3.1 error fires (the same applies to a literal example block ahead of the real trailing ones, §3.3.1).
- **Empty section content** — empty assistant turns are sent as `content: null`; other non-system roles send an empty string.
- **Trailing whitespace** on header lines, and CRLF line endings in general.
