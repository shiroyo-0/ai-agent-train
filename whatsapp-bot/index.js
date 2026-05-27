const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const http = require('http');

const API = 'http://127.0.0.1:8080';
const BOT_NAME = 'Shiro Nb.1.0';

const activeChats = new Set();
const processing = new Set(); // anti-loop
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
    const data = JSON.stringify({ message, session_id: `wa_${chatId}` });
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

client.on('message_create', async msg => {
    try {
        const chatId = msg.from || msg.to;
        const text = (msg.body || '').trim();
        const isFromMe = msg.fromMe;

        console.log(`[MSG] from=${msg.from} fromMe=${isFromMe} text="${text.slice(0,30)}"`);

        // Anti-loop: skip bot's own responses
        if (isFromMe && !text.startsWith('.ai')) return;

        const chat = await msg.getChat();
        const realChatId = chat.id._serialized;

        // Owner check
        const senderIsOwner = isFromMe || msg.from === ownerNumber;

        // Commands
        if (text.toLowerCase() === '.ai on' && senderIsOwner) {
            activeChats.add(realChatId);
            await chat.sendMessage(`🤖 *${BOT_NAME}* activated!`);
            console.log(`[ON] ${realChatId}`);
            return;
        }
        if (text.toLowerCase() === '.ai off' && senderIsOwner) {
            activeChats.delete(realChatId);
            await chat.sendMessage(`🤖 *${BOT_NAME}* deactivated.`);
            return;
        }
        if (text.toLowerCase() === '.ai status' && senderIsOwner) {
            try {
                const res = await fetch(`${API}/training/status`);
                const d = await res.json();
                await chat.sendMessage(`📊 Cycles: ${d.cycles} | Score: ${d.latest?.avg_score}/10`);
            } catch(e) { await chat.sendMessage('⚠️ Offline'); }
            return;
        }

        // Skip if not activated or is a command
        if (text.startsWith('.')) return;
        if (!activeChats.has(realChatId)) return;
        if (isFromMe) return; // don't respond to own non-command messages

        // Anti-duplicate
        const msgId = msg.id._serialized;
        if (processing.has(msgId)) return;
        processing.add(msgId);
        setTimeout(() => processing.delete(msgId), 30000);

        // Respond
        console.log(`[AI] Responding to: "${text.slice(0,50)}"`);
        await chat.sendStateTyping();
        const response = await askShiro(text, realChatId);
        await chat.sendMessage(response);
    } catch(e) {
        console.error('[ERR]', e.message);
    }
});

client.initialize();
