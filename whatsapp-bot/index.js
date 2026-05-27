const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const http = require('http');

const API = 'http://127.0.0.1:8080';
const BOT_NAME = 'Shiro Nb.1.0';

const activeChats = new Set();
let ownerNumber = null;

const client = new Client({
    authStrategy: new LocalAuth({ dataPath: './session' }),
    puppeteer: { headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'], executablePath: '/usr/bin/chromium-browser' }
});

client.on('qr', qr => { console.log('Scan QR:'); qrcode.generate(qr, { small: true }); });
client.on('ready', () => {
    ownerNumber = client.info.wid._serialized;
    console.log(`🤖 ${BOT_NAME} ready! Owner: ${ownerNumber}`);
});

function askShiro(message, chatId) {
    const data = JSON.stringify({ message, session_id: `wa_${chatId}`, force_cloud: true });
    return new Promise((resolve, reject) => {
        const req = http.request(`${API}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
            timeout: 60000,
        }, res => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => { try { resolve(JSON.parse(body).response); } catch(e) { reject(e); } });
        });
        req.on('error', reject);
        req.write(data);
        req.end();
    });
}

// Only listen to incoming messages (NOT own sent messages)
client.on('message', async msg => {
    const chat = await msg.getChat();
    const chatId = chat.id._serialized;
    const text = msg.body.trim().toLowerCase();
    const senderIsOwner = msg.from === ownerNumber;

    // Commands - owner only
    if (text === '.ai on' && senderIsOwner) {
        activeChats.add(chatId);
        await chat.sendMessage(`🤖 *${BOT_NAME}* activated!`);
        return;
    }
    if (text === '.ai off' && senderIsOwner) {
        activeChats.delete(chatId);
        await chat.sendMessage(`🤖 *${BOT_NAME}* deactivated.`);
        return;
    }
    if (text === '.ai status' && senderIsOwner) {
        try {
            const res = await fetch(`${API}/training/status`);
            const d = await res.json();
            await chat.sendMessage(`📊 Cycles: ${d.cycles} | Score: ${d.latest?.avg_score}/10`);
        } catch(e) { await chat.sendMessage('⚠️ Offline'); }
        return;
    }

    if (text.startsWith('.')) return;
    if (!activeChats.has(chatId)) return;

    // Respond
    try {
        await chat.sendStateTyping();
        const response = await askShiro(msg.body, chatId);
        await chat.sendMessage(response);
    } catch(e) {
        await chat.sendMessage(`⚠️ ${e.message}`);
    }
});

client.initialize();
