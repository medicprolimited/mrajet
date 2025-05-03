const fs = require('fs');
const backendUrl = process.env.BACKEND_URL;
if (!backendUrl) {
  throw new Error('BACKEND_URL environment variable is not set');
}
const configContent = `window.BACKEND_URL = "${backendUrl}";`;
fs.writeFileSync('scripts/config.js', configContent);