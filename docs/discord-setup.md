# Discord setup: a private server and the council bot

How to prepare Discord for `council chat discord` (M16 part B), for someone who
has only used Discord as a member. It takes about 20 minutes, needs no
programming, and costs nothing. Everything happens in the Discord app (desktop
or browser) and on the Discord developer site.

> **Status**: the bot is built (M16 part B, 2026-10-25); the keys in step 9 and
> the permissions in step 6 are what it uses. `council doctor --chat` checks them.

> **Privacy**: Discord is not end-to-end encrypted. Whatever is written in the
> channel (questions, debates, people's figures) is stored on Discord's servers.
> Use chat councils for topics you'd be comfortable discussing in a group chat;
> keep personal ones in the local TUI.

## 1. A few words of vocabulary

- **Server** (called a *guild* in the developer docs): a group space with its own
  members and channels. Creating one is free; you are its owner.
- **Channel**: a chat room inside a server (`#general`). A channel can be
  **private**: only the members and roles you add can see it.
- **Forum**: a channel made of **posts**, each with a title and its own
  conversation, like topics on a message board. The bot opens one post per
  question.
- **Thread**: a sub-conversation hanging off a text channel's message (a forum
  post is a thread too). With a text channel, the bot opens one thread per
  question.
- **Invite link**: the only way into a server. A server is **not** public unless
  its owner turns on *Community* and *Server Discovery*: without an invite, nobody
  can find or join it.
- **Bot**: an account driven by a program (here: `council`, running on your PC).
  It belongs to an **application** you create on the developer site, and logs in
  with a secret **token**.
- **Webhook**: a way to post messages under any name and avatar. The bot uses one
  so each persona appears as its own speaker.
- **ID**: every server, channel and user has a long number (an ID). The bot's
  config uses IDs, not names, because names can change.

## 2. Is my existing server public?

If you already have a server (even an old one where you're alone), check it:

1. Open the server, click its name (top left) → **Server Settings**.
2. In the left menu, look for **Enable Community** (or a **Community** section).
   - If it offers to *enable* Community, Community is off: the server can't be
     discovered, and people need an invite.
   - If Community is on, check **Discovery**: it must be off.
3. **Server Settings → Invites** lists every invite link still valid. Revoke the
   ones you don't need (the cross next to each).

Recommendation: create a **new server** dedicated to the council (step 3). It
keeps the bot and its permissions away from anything else, and you can delete
the whole thing at any time.

## 3. Create a private test server

1. In the left column of the Discord app, click **+** (*Add a Server*).
2. **Create My Own** → **For me and my friends**.
3. Name it (e.g. "Council"), click the upload circle to set its icon to
   `docs/assets/council-logo.png` (the project logo), and click **Create**. (Later:
   Server Settings → **Overview** → icon.)
4. Create the channel the bot will use, as a **forum**: next to **TEXT
   CHANNELS**, click **+**.
   - Channel type: **Forum**. Name: `council`.
   - Turn on **Private Channel**, then **Next** and **Create Channel** (you can
     add people later).
5. Create the **lobby** post, where you type commands: open `#council` → **New
   Post** → title `lobby`, any first message (e.g. "Commands here") → **Post**.
   Then right-click the post → **Pin Post**, so it stays at the top.

Why a forum: each debate becomes its own post, titled with the question, so past
debates are easy to find and reopen. Why a lobby: a forum has no message box of
its own, only its posts do, so `/ask` needs a post to be typed in. Commands also
work inside a debate's post; chat lines in the lobby are ignored.

> **A text channel works too.** If you'd rather (or if **Forum** isn't offered
> as a channel type), pick **Text** instead and skip the lobby: commands go in
> `#council` itself, and each debate is a thread under it.

Only you can see `#council` for now. Friends you invite to the server will see
`#general` but not `#council` until you add them (step 8).

## 4. Turn on Developer Mode (to copy IDs)

1. Click the gear icon next to your name (bottom left) → **User Settings**.
2. **Advanced** (under *App Settings*) → turn on **Developer Mode**.

From now on, a right-click (long-press on mobile) shows **Copy ID**:

- **Server ID**: right-click the server icon in the left column → **Copy Server ID**.
- **Channel ID**: right-click `#council` → **Copy Channel ID**.
- **Your user ID**: right-click your name in the member list → **Copy User ID**.
- A **friend's user ID**: same, once they've joined.

Keep these numbers for the config (step 9). IDs are not secret, but they are
personal identifiers: keep them in `config.local.toml`, never in a committed file.

## 5. Create the bot (developer site)

1. Go to <https://discord.com/developers/applications> and log in with your
   usual Discord account. On a first visit Discord asks "What brings you to the
   Developer Portal?": pick **Build a Bot** → **Continue to Portal** (or
   **Skip**; it only tailors the portal's pages).
2. **New Application** → name it (e.g. "Council"; friends will see this name) →
   accept the terms → **Create**. Optional: **General Information** → **App Icon**
   → `docs/assets/council-logo.png`, so the bot wears the same logo.
3. **Installation** tab (left menu): set **Install Link** to **None**, untick
   **User Install** under *Installation Contexts* (keep **Guild Install**), then
   click **Save Changes** in the green bar at the bottom. Do this *before* the
   next step: Discord refuses to make a bot private while it has an install link
   ("Private application cannot have a default authorization link"), and the
   change only counts once saved.
4. **Bot** tab:
   - **Public Bot**: turn it **off**, so only you can add the bot to a server.
   - **Requires OAuth2 Code Grant**: leave it **off**.
   - **Privileged Gateway Intents**: turn on **Message Content Intent** (the bot
     must read what people write in the channel). Leave *Presence* and *Server
     Members* off.
   - **Save Changes**.
5. Still on the **Bot** tab: **Reset Token** → confirm → **Copy**. This is the
   bot's password: paste it straight into `config.local.toml` (step 9). Discord
   shows it only once; if you lose it, reset it again.

> **The token is a password.** Anyone with it can act as your bot in every
> server it's in. Never paste it in a chat, a screenshot, an issue or a committed
> file. If it leaks: **Bot → Reset Token** (the old one stops working at once),
> then update `config.local.toml`.

## 6. Add the bot to your server

1. On the developer site, **OAuth2** tab → **OAuth2 URL Generator**.
2. **Scopes**: tick **bot** and **applications.commands** (the slash commands
   such as `/run` and `/vote`).
3. **Bot Permissions** (appears below once *bot* is ticked), tick only:
   - **View Channels**
   - **Send Messages**
   - **Send Messages in Threads**
   - **Create Public Threads** (needed for a text channel; harmless for a forum)
   - **Read Message History**
   - **Manage Webhooks** (the bot makes one webhook per persona in `#council`, named
     `council-<persona>`, so each persona posts under its own name and avatar)
   - **Embed Links**
4. **Integration Type**: **Guild Install**.
5. Copy the **Generated URL** at the bottom, open it in your browser, choose your
   new server in the list → **Continue** → **Authorize**.

The bot now appears in the server's member list, offline (it comes online when
`council chat discord` runs on your PC).

## 7. Let the bot into the private channel

A private channel hides itself from the bot too.

1. Right-click `#council` → **Edit Channel** → **Permissions**.
2. Under the private channel settings, **Add members or roles** → pick the bot
   (by its name, e.g. "Council") → **Done**.

## 8. Invite friends

1. Right-click the server icon → **Invite People**.
2. Click **Edit invite link** (bottom): set **Expire After** to *1 day* and **Max
   Number of Uses** to the number of friends; leave *Grant temporary membership*
   off. **Generate a New Link**, then send it to them privately.
3. Once they've joined: `#council` → **Edit Channel → Permissions → Add members
   or roles** → add each friend.
4. Copy each friend's **User ID** (step 4) for the allowlist.

The bot answers only the user IDs listed in its config, in `#council` only:
someone who joins the server but isn't on the list is ignored, and so are direct
messages.

## 9. Put the settings in `config.local.toml`

`config.local.toml` is gitignored; never move these values to `config.toml`.

```toml
[chat.discord]
token = "paste-the-bot-token-here"
guild = 123456789012345678          # Server ID
channel = 123456789012345678        # #council's Channel ID (forum or text)
owner = 123456789012345678          # your User ID: starts questions, /run, /offline
allowed_users = [                   # friends' User IDs: speak and answer the Chair
  234567890123456789,
]
# Optional:
# council = "finance"               # the council /ask uses when none is given
# names = { "234567890123456789" = "Ben" }   # a nickname instead of the username
# question_timeout_s = 600          # how long the Chair's questions wait
# max_lines_per_round = 10          # per person, between two Chair updates
```

Then protect the file (it holds the token) and check the setup:

```bash
chmod 600 config.local.toml
uv run council doctor --chat     # token, intent, channel kind, permissions
uv run council chat discord      # start the bot; /offline in Discord stops it
```

**Names in the debate.** People are known by their user ID (the allowlist); the
name the council uses for them is their Discord **username** (not the display
name), with `.` turned into `-` (`ana.martin` → `@ana-martin`). It is fixed for the
whole debate, even if they rename themselves, and gets a suffix if it would clash
with yours or a persona's. `names` overrides it.

**Commands** go in the `lobby` post or in a debate's post (in a text channel:
`#council` or a debate's thread); replies are private to whoever typed them:

| Command | Who | What |
|---|---|---|
| `/ask question [council]` | owner | start a debate in a new post (one at a time); `council` is a dropdown of the councils, default `[chat.discord] council` |
| `/run [rounds]` | owner | run rounds (default: the rest); the Chair's questions wait for the person asked |
| `/pause`, `/vote`, `/close` | owner | pause at the next turn, final positions, the synthesis |
| `/offline`, `/resume` | owner | stop after the current turn and disconnect; reopen later |
| `/status`, `/flaws` | everyone listed | where the debate is; errors caught so far |

Everyone listed can write in the debate's thread: lines go to the council (up to
10 per person per round), and an answer to a question from the Chair counts as
the answer. Debates always run on local models only, with memory off; sessions are
saved in `data/chat/`.

## 10. Turning it off

- **Pause**: just don't run `council chat discord`; the bot shows offline.
- **Remove the bot from the server**: Server Settings → **Integrations** → the
  bot → **Remove**; or right-click the bot in the member list → **Kick**.
- **Kill the token**: developer site → **Bot → Reset Token**.
- **Delete everything**: developer site → your application → **General
  Information** → **Delete App**; and Server Settings → **Delete Server**.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Private application cannot have a default authorization link" when turning off *Public Bot* | Set **Installation → Install Link** to **None** first (step 5). |
| The bot is online but never answers | **Message Content Intent** is off (step 5), or the bot can't see `#council` (step 7), or your user ID isn't the `owner`/`allowed_users` value. |
| The authorize link shows a white or black page, then "Opening Discord App" and nothing | The browser is handing the link to the desktop app. Log in at discord.com in the browser, paste the link in a new tab there and cancel any "Open Discord?" prompt (or use a private window). |
| Your server isn't in the list when authorizing the bot | You must own the server, or have *Manage Server* on it. |
| "Use the council's commands in its channel." | The command was typed outside `#council`: use the `lobby` post (forum) or `#council` (text channel). |
| Personas post without their avatar | A persona webhook couldn't be made (permission or Discord's 15-webhooks limit): the bot falls back to its shared `council` webhook and logs a warning. Deleting the `council-*` webhooks (Server Settings → Integrations → Webhooks) is harmless; the bot recreates them. |
| Slash commands don't appear | The **applications.commands** scope was missing: generate the URL again with it (step 6) and reopen it; Discord may also take a minute to show new commands. |
| A friend can't see `#council` | Add them to the channel's permissions (step 8). |
| "Missing Permissions" in the bot's log | A permission from step 6 wasn't ticked: regenerate the URL with it and authorize again (the bot keeps its place). |
