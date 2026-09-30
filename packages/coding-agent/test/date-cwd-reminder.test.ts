import { afterEach, describe, expect, it } from "bun:test";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import { invalidateMessageCache } from "@oh-my-pi/pi-agent-core/compaction/message-cache";
import type { Api, Context, Message, Model, ModelSpec, UserMessage } from "@oh-my-pi/pi-ai";
import { clearCustomApis, registerCustomApi } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { createAgentSession } from "@oh-my-pi/pi-coding-agent/sdk";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { DateCwdReminderInjector, renderDateCwdReminder } from "@oh-my-pi/pi-coding-agent/session/date-cwd-reminder";
import { convertToLlm, stripImagesFromMessage, wrapSteeringForModel } from "@oh-my-pi/pi-coding-agent/session/messages";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { formatLocalCalendarDate } from "@oh-my-pi/pi-tui/chrome/local-date";
import { normalizePromptPath } from "@oh-my-pi/pi-coding-agent/utils/prompt-path";
import { TempDir } from "@oh-my-pi/pi-utils";
import { COLLAB_PROMPT_MESSAGE_TYPE } from "@oh-my-pi/pi-wire";
import { createAssistantMessage } from "./helpers/agent-session-setup";

describe("date-cwd-reminder", () => {
	afterEach(() => {
		clearCustomApis();
	});

	describe("DateCwdReminderInjector", () => {
		it("injects the first reminder without mutating the context", () => {
			const systemPrompt = ["PROJECT\n<critical>\n- Must act.\n</critical>"];
			const messages: Message[] = [{ role: "user", content: "hello", timestamp: 1 }, createAssistantMessage("hi")];
			const context: Context = { systemPrompt, messages };
			const injector = new DateCwdReminderInjector();

			const out = injector.transform(context, "2026-08-14", "/work/omp");

			expect(out).not.toBe(context);
			expect(out.systemPrompt).toBe(systemPrompt);
			expect(out.messages).not.toBe(messages);
			expect(out.messages[0]).toEqual({
				role: "user",
				content: `${renderDateCwdReminder("2026-08-14", "/work/omp")}\n\nhello`,
				timestamp: 1,
			});
			expect(out.messages[1]).toBe(messages[1]);
			expect(context.messages).toBe(messages);
		});

		it("prepends a text part before image parts", () => {
			const context: Context = {
				systemPrompt: ["system"],
				messages: [
					{
						role: "user",
						content: [{ type: "image", data: "img", mimeType: "image/png" }],
						timestamp: 1,
					},
				],
			};

			const out = new DateCwdReminderInjector().transform(context, "2026-08-14", "/work/omp");

			expect(out.messages[0]?.content).toEqual([
				{ type: "text", text: renderDateCwdReminder("2026-08-14", "/work/omp") },
				{ type: "image", data: "img", mimeType: "image/png" },
			]);
		});

		it("leaves contexts without a system prompt or user message untouched", () => {
			const injector = new DateCwdReminderInjector();
			const noSystem: Context = {
				systemPrompt: [],
				messages: [{ role: "user", content: "hi", timestamp: 1 }],
			};
			const noUser: Context = { systemPrompt: ["system"], messages: [createAssistantMessage("hi")] };

			expect(injector.transform(noSystem, "2026-08-14", "/cwd")).toBe(noSystem);
			expect(injector.transform(noUser, "2026-08-14", "/cwd")).toBe(noUser);
		});

		it("keeps prior reminder bytes and moves a changed reminder to the next user turn", () => {
			const injector = new DateCwdReminderInjector();
			const firstUser: Message = { role: "user", content: "first", timestamp: 1 };
			const firstContext: Context = { systemPrompt: ["system"], messages: [firstUser] };

			const first = injector.transform(firstContext, "2026-08-14", "/old");
			const firstInjected = first.messages[0]!;
			const secondUser: Message = { role: "user", content: "second", timestamp: 2 };
			const second = injector.transform(
				{
					systemPrompt: firstContext.systemPrompt,
					messages: [firstUser, createAssistantMessage("done"), secondUser],
				},
				"2026-08-15",
				"/new",
			);

			expect(second.messages[0]).toBe(firstInjected);
			expect(second.messages[0]?.content).toBe(firstInjected.content);
			expect(second.messages[2]?.content).toBe(`${renderDateCwdReminder("2026-08-15", "/new")}\n\nsecond`);
			expect(firstUser.content).toBe("first");
			expect(secondUser.content).toBe("second");
		});

		it("reuses injected message objects on provider request replay", () => {
			const injector = new DateCwdReminderInjector();
			const firstUser: Message = { role: "user", content: "first", timestamp: 1 };
			const context: Context = { systemPrompt: ["system"], messages: [firstUser] };

			const first = injector.transform(context, "2026-08-14", "/work/omp");
			const replay = injector.transform({ ...context, messages: [...context.messages] }, "2026-08-14", "/work/omp");

			expect(replay.messages[0]).toBe(first.messages[0]);
		});
	});
});

function steeringMessage(kind: "user" | "collab", content: UserMessage["content"], timestamp = 1) {
	return kind === "user"
		? { role: "user" as const, content, steering: true, timestamp }
		: {
				role: "custom" as const,
				customType: COLLAB_PROMPT_MESSAGE_TYPE,
				content,
				display: true,
				attribution: "user" as const,
				timestamp,
			};
}

function steeringRequest(
	injector: DateCwdReminderInjector,
	messages: AgentMessage[],
	date: string,
	cwd: string,
): Message[] {
	return injector.transform(
		{ systemPrompt: ["system"], messages: convertToLlm(wrapSteeringForModel(messages)) },
		date,
		cwd,
	).messages;
}

function reminderText(message: Message): string {
	if (typeof message.content === "string") return message.content;
	const text: string[] = [];
	for (const part of message.content) {
		if (part.type === "text" && "text" in part && typeof part.text === "string") text.push(part.text);
	}
	return text.join("\n");
}

describe("steering date/cwd cache stability", () => {
	for (const kind of ["user", "collab"] as const) {
		describe(kind, () => {
			it("preserves historical bytes across new turns, date/cwd changes, and A-B-A replay", () => {
				const injector = new DateCwdReminderInjector();
				const root = steeringMessage(kind, "first steer");
				const history: AgentMessage[] = [root];
				const first = steeringRequest(injector, history, "2026-08-14", "/old");
				const firstBytes = JSON.stringify(first);
				expect(reminderText(first[0]!)).toContain(renderDateCwdReminder("2026-08-14", "/old"));
				expect(steeringRequest(injector, history, "2026-08-14", "/old")[0]).toBe(first[0]);

				history.push(createAssistantMessage("done"), steeringMessage(kind, "same-day steer", 2));
				const sameDay = steeringRequest(injector, history, "2026-08-14", "/old");
				expect(sameDay[0]).toBe(first[0]);
				expect(reminderText(sameDay[2]!)).not.toContain("<system-reminder>");
				const sameDayBytes = JSON.stringify(sameDay);

				history.push(steeringMessage(kind, "next-day steer", 3));
				const nextDay = steeringRequest(injector, history, "2026-08-15", "/new");
				expect(JSON.stringify(nextDay.slice(0, 3))).toBe(sameDayBytes);
				expect(reminderText(nextDay[3]!)).toContain(renderDateCwdReminder("2026-08-15", "/new"));
				const nextDayBytes = JSON.stringify(nextDay);

				const cwdChange = steeringRequest(injector, history, "2026-08-15", "/elsewhere");
				expect(JSON.stringify(cwdChange.slice(0, 4))).toBe(nextDayBytes);
				expect(cwdChange[4]).toMatchObject({
					role: "developer",
					content: renderDateCwdReminder("2026-08-15", "/elsewhere"),
				});
				const back = steeringRequest(injector, history, "2026-08-14", "/old");
				expect(back.slice(0, 5)).toEqual(cwdChange);
				expect(back[5]).toMatchObject({ role: "developer", content: renderDateCwdReminder("2026-08-14", "/old") });
				expect(JSON.stringify(back.slice(0, 1))).toBe(firstBytes);
				expect(steeringRequest(injector, history, "2026-08-14", "/old")).toEqual(back);
				expect(root.content).toBe("first steer");
			});

			it("refreshes owner edits and image removal on first and later reminder carriers", () => {
				const injector = new DateCwdReminderInjector();
				const text = { type: "text" as const, text: "first steer" };
				const image = { type: "image" as const, data: "aW1n", mimeType: "image/png" };
				const root = steeringMessage(kind, [text, image]);
				const history: AgentMessage[] = [root];
				steeringRequest(injector, history, "2026-08-14", "/old");
				text.text = "edited first steer";
				invalidateMessageCache(root);
				const editedRoot = steeringRequest(injector, history, "2026-08-14", "/old");
				expect(reminderText(editedRoot[0]!)).toContain("edited first steer");
				expect(reminderText(editedRoot[0]!)).toContain(renderDateCwdReminder("2026-08-14", "/old"));
				expect(editedRoot[0]!.content).toContainEqual(image);
				expect(stripImagesFromMessage(root)).toBe(1);
				const strippedRoot = steeringRequest(injector, history, "2026-08-14", "/old");
				expect(strippedRoot[0]!.content).not.toContainEqual(image);
				expect(reminderText(strippedRoot[0]!)).toContain("edited first steer");
				expect(reminderText(strippedRoot[0]!)).toContain(renderDateCwdReminder("2026-08-14", "/old"));
				const rootBytes = JSON.stringify(strippedRoot);

				const later = steeringMessage(kind, "later steer", 2);
				history.push(later);
				steeringRequest(injector, history, "2026-08-15", "/new");
				later.content = "edited later steer";
				invalidateMessageCache(later);
				const editedLater = steeringRequest(injector, history, "2026-08-15", "/new");
				expect(JSON.stringify(editedLater.slice(0, 1))).toBe(rootBytes);
				expect(reminderText(editedLater[1]!)).toContain("edited later steer");
				expect(reminderText(editedLater[1]!)).toContain(renderDateCwdReminder("2026-08-15", "/new"));

				later.content = [{ type: "text", text: "later image" }, image];
				invalidateMessageCache(later);
				const withImage = steeringRequest(injector, history, "2026-08-15", "/new");
				expect(withImage[1]!.content).toContainEqual(image);
				expect(stripImagesFromMessage(later)).toBe(1);
				const strippedLater = steeringRequest(injector, history, "2026-08-15", "/new");
				expect(JSON.stringify(strippedLater.slice(0, 1))).toBe(rootBytes);
				expect(strippedLater[1]!.content).not.toContainEqual(image);
				expect(reminderText(strippedLater[1]!)).toContain("later image");
				expect(reminderText(strippedLater[1]!)).toContain(renderDateCwdReminder("2026-08-15", "/new"));
				expect(steeringRequest(injector, history, "2026-08-15", "/new")).toEqual(strippedLater);
			});

			it("restores reminders after side requests and tail trimming without growing on replay", () => {
				const injector = new DateCwdReminderInjector();
				const root = steeringMessage(kind, "main history");
				const history: AgentMessage[] = [root, createAssistantMessage("done")];
				const first = steeringRequest(injector, history, "2026-08-14", "/old");
				const firstBytes = JSON.stringify(first);
				const sideHistory: AgentMessage[] = [...history, { role: "user", content: "temporary", timestamp: 2 }];
				const side = steeringRequest(injector, sideHistory, "2026-08-15", "/new");
				expect(JSON.stringify(side.slice(0, 2))).toBe(firstBytes);
				expect(reminderText(side[2]!)).toBe(`${renderDateCwdReminder("2026-08-15", "/new")}\n\ntemporary`);
				const main = steeringRequest(injector, history, "2026-08-15", "/new");
				expect(JSON.stringify(main.slice(0, 2))).toBe(firstBytes);
				expect(main[2]).toMatchObject({ role: "developer", content: renderDateCwdReminder("2026-08-15", "/new") });
				for (let replay = 0; replay < 3; replay++) {
					expect(steeringRequest(injector, sideHistory, "2026-08-15", "/new")).toEqual([...main, side[2]!]);
					expect(steeringRequest(injector, history, "2026-08-15", "/new")).toEqual(main);
				}

				// Removing the developer control's anchor must recover too, not just
				// removing an injected user carrier as the side request did above.
				const trimmed = steeringRequest(injector, [root], "2026-08-15", "/new");
				expect(trimmed[0]).toBe(first[0]);
				expect(trimmed[1]).toMatchObject({
					role: "developer",
					content: renderDateCwdReminder("2026-08-15", "/new"),
				});
				expect(steeringRequest(injector, [root], "2026-08-15", "/new")).toEqual(trimmed);
				expect(steeringRequest(injector, history, "2026-08-15", "/new")).toEqual([...trimmed, first[1]!, main[2]!]);
			});
		});
	}

	it("isolates injectors and resets when an equal-content root is replaced", () => {
		const one = new DateCwdReminderInjector();
		const two = new DateCwdReminderInjector();
		const root = steeringMessage("user", "same content");
		const first = steeringRequest(one, [root], "2026-08-14", "/one");
		const other = steeringRequest(two, [root], "2026-08-15", "/two");
		expect(reminderText(other[0]!)).toContain(renderDateCwdReminder("2026-08-15", "/two"));
		expect(reminderText(other[0]!)).not.toContain(renderDateCwdReminder("2026-08-14", "/one"));
		expect(steeringRequest(one, [root], "2026-08-14", "/one")[0]).toBe(first[0]);
		const replacement = steeringMessage("user", "same content");
		const replaced = steeringRequest(one, [replacement], "2026-08-15", "/new");
		expect(replaced).toHaveLength(1);
		expect(reminderText(replaced[0]!)).toContain(renderDateCwdReminder("2026-08-15", "/new"));
		expect(reminderText(replaced[0]!)).not.toContain(renderDateCwdReminder("2026-08-14", "/one"));
		expect(steeringRequest(one, [replacement], "2026-08-15", "/new")[0]).toBe(replaced[0]);
		expect(steeringRequest(two, [root], "2026-08-15", "/two")[0]).toBe(other[0]);
	});
});

describe("date-cwd reminder on the provider wire", () => {
	const sessions: Array<{ dispose(): Promise<void> }> = [];

	afterEach(async () => {
		clearCustomApis();
		for (const session of sessions.splice(0)) {
			await session.dispose();
		}
	});

	it("keeps the date/cwd out of the system prompt and pins the reminder to the first user turn across requests", async () => {
		using tempDir = TempDir.createSync("@pi-date-cwd-reminder-");
		const api = "test-date-cwd-reminder";
		const contexts: Context[] = [];
		registerCustomApi(api, (_model, context) => {
			contexts.push(context);
			const stream = new AssistantMessageEventStream();
			queueMicrotask(() => {
				const message = createAssistantMessage("ok");
				stream.push({ type: "text_delta", contentIndex: 0, delta: "ok", partial: message });
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		});
		const model = buildModel({
			id: "date-cwd-reminder",
			name: "Date cwd reminder",
			api,
			provider: "managed-primary",
			baseUrl: "http://127.0.0.1:8080/v1",
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 4096,
			maxTokens: 1024,
		} as ModelSpec<Api>) as Model<Api>;
		const authStorage = await AuthStorage.create(tempDir.join("auth.db"));
		authStorage.keys.setRuntime(model.provider, "test-key");
		const modelRegistry = new ModelRegistry(authStorage, tempDir.join("models.yml"));
		const { session } = await createAgentSession({
			cwd: tempDir.path(),
			agentDir: tempDir.path(),
			sessionManager: SessionManager.inMemory(tempDir.path()),
			authStorage,
			modelRegistry,
			settings: Settings.isolated({ "compaction.enabled": false }),
			model,
			disableExtensionDiscovery: true,
			skills: [],
			contextFiles: [],
			promptTemplates: [],
			slashCommands: [],
			enableMCP: false,
			enableLsp: false,
			skipPythonPreflight: true,
			taskDepth: 1,
			agentId: "SubAgent",
		});
		sessions.push(session);

		try {
			await session.sendUserMessage("first");

			expect(contexts).toHaveLength(1);
			// The volatile line must no longer live in the system prompt: open-weight
			// chat templates render tool schemas after the system content, so any
			// per-request byte there invalidates the whole tool-schema cache (#7404).
			const systemPrompt = contexts[0]!.systemPrompt?.join("\n") ?? "";
			expect(systemPrompt).not.toContain("Today");
			expect(systemPrompt).not.toContain("current working directory");
			expect(systemPrompt).not.toContain(formatLocalCalendarDate());

			const firstUser = contexts[0]!.messages[0]!;
			expect(firstUser.role).toBe("user");
			const firstText =
				typeof firstUser.content === "string" ? firstUser.content : JSON.stringify(firstUser.content);
			expect(firstText).toContain("<system-reminder>");
			expect(firstText).toContain(formatLocalCalendarDate());
			expect(firstText).toContain(normalizePromptPath(tempDir.path()));

			// A second request must re-emit byte-identical reminder bytes so the
			// conversation prefix (system + tools + first turn) stays cached.
			await session.sendUserMessage("second");
			expect(contexts).toHaveLength(2);
			const secondFirst = contexts[1]!.messages[0]!;
			expect(secondFirst.role).toBe("user");
			expect(typeof secondFirst.content).toBe(typeof firstUser.content);
			expect(secondFirst.content).toEqual(firstUser.content);
		} finally {
			authStorage.close();
		}
	});
});
