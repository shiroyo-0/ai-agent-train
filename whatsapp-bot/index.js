const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const http = require('http');

const API = 'http://localhost:8080';
const BOT_NAME = 'Shiro Nb.1.0';
const FOOTER = '\n\n_⚡ Powered by Shiro Nb.1.0 AI_';

// Track which chats have AI enabled
const activeChats = new Set();

const client = new Client({
    authStrategy: new LocalAuth({ dataPath: './session' }),
    puppeteer: { headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'], executablePath: '/usr/bin/chromium-browser' }
});

client.on('qr', qr => {
    console.log('Scan QR code:');
    qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
    console.log(`🤖 ${BOT_NAME} WhatsApp Bot ready!`);
});

// Call Shiro API - force cloud (fast)
async function askShiro(message, chatId) {
    const data = JSON.stringify({ message, session_id: `wa_${chatId}`, force_cloud: true });
    return new Promise((resolve, reject) => {
        const req = http.request(`${API}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
            timeout: 60000,
        }, res => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                try { resolve(JSON.parse(body).response); }
                catch (e) { reject(e); }
            });
        });
        req.on('error', reject);
        req.write(data);
        req.end();
    });
}

client.on('message', async msg => {
    const chat = await msg.getChat();
    const chatId = chat.id._serialized;
    const text = msg.body.trim().toLowerCase();

    // Commands (always active)
    if (text === '.ai on') {
        activeChats.add(chatId);
        await msg.reply(`🤖 *${BOT_NAME}* activated!\nSend any message and I'll respond.\nType *.ai off* to disable.${FOOTER}`);
        return;
    }

    if (text === '.ai off') {
        activeChats.delete(chatId);
        await msg.reply(`🤖 *${BOT_NAME}* deactivated.\nType *.ai on* to enable again.`);
        return;
    }

    if (text === '.ai status') {
        try {
            const res = await fetch(`${API}/training/status`);
            const d = await res.json();
            const info = d.cycles
                ? `📊 *Training Status*\nCycles: ${d.cycles}\nLast score: ${d.latest.avg_score}/10\nHQ examples: ${d.latest.high_quality}/${d.latest.examples}`
                : 'No training data yet.';
            await msg.reply(`🤖 *${BOT_NAME}*\n${info}${FOOTER}`);
        } catch (e) {
            await msg.reply(`⚠️ Server offline`);
        }
        return;
    }

    // Only respond if AI is enabled for this chat
    if (!activeChats.has(chatId)) return;

    // Skip if message is from bot itself
    if (msg.fromMe) return;

    // Respond with AI
    try {
        await chat.sendStateTyping();
        const response = await askShiro(msg.body, chatId);
        await msg.reply(`${response}${FOOTER}`);
    } catch (e) {
        await msg.reply(`⚠️ Error: ${e.message}`);
    }
});

client.initialize();
