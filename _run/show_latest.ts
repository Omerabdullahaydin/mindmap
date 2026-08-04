// ============================================================================
// BU DOSYA NE İŞE YARAR?
// ----------------------------------------------------------------------------
// "npm run show" komutunun çalıştırdığı script. mindmaps/ klasöründeki EN SON
// değiştirilen (en güncel) .html dosyasını bulup varsayılan tarayıcıda açar —
// pipeline'ı çalıştırdıktan sonra elle dosya yolunu kopyalayıp tarayıcıya
// yapıştırmak zorunda kalmamak için.
// ============================================================================
import { readdirSync, statSync } from "fs";
import path from "path";
import { exec } from "child_process";

const mindmapsDir = path.join(process.cwd(), "mindmaps");

const htmlFiles = readdirSync(mindmapsDir)
  .filter(f => f.toLowerCase().endsWith(".html"))
  .map(f => {
    const fullPath = path.join(mindmapsDir, f);
    return { fullPath, mtime: statSync(fullPath).mtimeMs };
  });

if (htmlFiles.length === 0) {
  console.error(`"${mindmapsDir}" klasöründe hiç HTML dosyası bulunamadı. Önce "npm start" ile bir mindmap üretin.`);
  process.exit(1);
}

const latest = htmlFiles.sort((a, b) => b.mtime - a.mtime)[0];

console.log(`En son üretilen mindmap açılıyor: ${latest.fullPath}`);

// Platforma göre "dosyayı varsayılan uygulamayla aç" komutu farklı.
const openCommand =
  process.platform === "win32" ? `start "" "${latest.fullPath}"` :
  process.platform === "darwin" ? `open "${latest.fullPath}"` :
  `xdg-open "${latest.fullPath}"`;

exec(openCommand, (error) => {
  if (error) {
    console.error("Tarayıcı açılamadı:", error.message);
    console.log(`Dosyayı elle açmak isterseniz: ${latest.fullPath}`);
  }
});
