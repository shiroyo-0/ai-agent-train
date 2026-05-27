const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const http = require('http');

const API = 'http://localhost:8080';
const BOT_NAME = 'Shiro Nb.1.0';

// Track which chats have AI enabled
const activeChats = new Set();
let ownerNumber = null;

const client = new Client({
    authStrategy: new LocalAuth({ dataPath: './session' }),
    puppeteer: { headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'], executablePath: '/usr/bin/chromium-browser' }
});

client.on('qr', qr => {
    console.log('Scan QR code:');
    qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
    ownerNumber = client.info.wid._serialized;
    console.log(`🤖 ${BOT_NAME} WhatsApp Bot ready!`);
    console.log(`👤 Owner: ${ownerNumber}`);
});

// Call Shiro API
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

function isOwner(msg) {
    return msg.from === ownerNumber || msg.fromMe;
}

client.on('message_create', async msg => {
    const chat = await msg.getChat();
    const chatId = chat.id._serialized;
    const text = msg.body.trim().toLowerCase();

    // Management commands - OWNER ONLY
    if (text === '.ai on') {
        if (!isOwner(msg)) return;
        activeChats.add(chatId);
        await chat.sendMessage(`🤖 *${BOT_NAME}* activated!`);
        return;
    }
    if (text === '.ai off') {
        if (!isOwner(msg)) return;
        activeChats.delete(chatId);
        await chat.sendMessage(`🤖 *${BOT_NAME}* deactivated.`);
        return;
    }
    if (text === '.ai status') {
        if (!isOwner(msg)) return;
        try {
            const res = await fetch(`${API}/training/status`);
            const d = await res.json();
            await chat.sendMessage(`📊 Cycles: ${d.cycles} | Score: ${d.latest?.avg_score}/10`);
        } catch (e) { await chat.sendMessage(`⚠️ Server offline`); }
        return;
    }

    // Skip commands
    if (text.startsWith('.')) return;

    // Only respond if AI is enabled for this chat
    if (!activeChats.has(chatId)) return;

    // Skip own management messages, but respond to own questions
    if (msg.fromMe && text.length < 3) return;

    // Respond to ALL messages (including own)
    try {
        await chat.sendStateTyping();
        const response = await askShiro(msg.body, chatId);
        await chat.sendMessage(response);
    } catch (e) {
        await chat.sendMessage(`⚠️ Error: ${e.message}`);
    }
});

client.initialize();
