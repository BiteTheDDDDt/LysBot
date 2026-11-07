"""Export a read-only, offline HTML gallery from the running bot's meme DB."""
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bot 表情图库</title>
<style>
:root{color-scheme:light;--ink:#173a42;--muted:#607c82;--line:#cde0df}*{box-sizing:border-box}body{margin:0;background:linear-gradient(130deg,#e7f5ef,#e8effa 65%,#f5ecf7);color:var(--ink);font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}header{padding:36px max(24px,4vw) 22px}h1{margin:0;font-size:32px;letter-spacing:2px}header p{color:var(--muted);margin:6px 0}.toolbar{position:sticky;top:0;z-index:2;padding:14px max(24px,4vw);background:#f6fbf6eb;backdrop-filter:blur(18px);border-block:1px solid var(--line);display:flex;flex-wrap:wrap;gap:10px;align-items:center}input,select,button{font:inherit;color:inherit;padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:#fff}input{flex:1;min-width:220px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}.count{color:var(--muted)}main{padding:24px max(24px,4vw);display:grid;grid-template-columns:repeat(auto-fill,minmax(285px,1fr));gap:20px;align-items:start}.card{overflow:hidden;border:1px solid #fff;border-radius:18px;background:#fffffff0;box-shadow:0 8px 28px #274c5310}.picture{height:260px;background:repeating-conic-gradient(#edf1f3 0% 25%,#f8fafb 0% 50%) 0/20px 20px;display:flex;align-items:center;justify-content:center}.picture img{display:block;width:100%;height:100%;object-fit:contain;cursor:zoom-in}.body{padding:16px}.badges{display:flex;flex-wrap:wrap;gap:7px;align-items:center}.badge{font-size:12px;padding:3px 9px;border-radius:20px;background:#e4f1ed}.off{background:#f6e7df}.unknown{background:#eee}.description{white-space:pre-wrap;overflow-wrap:anywhere;margin:12px 0}.meta{color:var(--muted);font-size:12px}details{margin-top:8px}summary{cursor:pointer;color:var(--muted)}footer{padding:10px 24px 40px;display:flex;justify-content:center;align-items:center;gap:16px}dialog{border:0;border-radius:16px;padding:16px;max-width:95vw;max-height:95vh;background:#f7fafb}dialog::backdrop{background:#102b39c9}dialog img{display:block;max-width:88vw;max-height:76vh;object-fit:contain}dialog .actions{display:flex;justify-content:space-between;align-items:center;gap:16px;padding-top:10px}a{color:#226c8d}.empty{grid-column:1/-1;padding:50px;text-align:center;color:var(--muted)}
</style>
<header><h1>Bot 表情图库</h1><p id="summary"></p><p>完整图片与入库描述 · 点击图片放大 · 本页为只读快照</p></header>
<div class="toolbar"><input id="query" type="search" placeholder="搜索描述、图片 ID 或哈希" aria-label="搜索"><select id="topic" aria-label="情绪分类"><option value="">全部分类</option></select><select id="kind" aria-label="图片用途"><option value="">全部用途</option><option value="reaction">情绪反应图</option><option value="joke">网络梗图</option><option value="unclassified">非梗图 / 未确认</option></select><select id="eligible" aria-label="发送状态"><option value="">全部发送状态</option><option value="1">允许随机发送</option><option value="0">禁止随机发送</option><option value="null">尚未标记</option></select><select id="sort" aria-label="排序"><option value="new">最新收录</option><option value="old">最早收录</option><option value="used">使用次数最多</option></select><span class="count" id="count"></span></div>
<main id="grid"></main><footer><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></footer>
<dialog id="viewer"><img id="large" alt="表情原图"><div class="actions"><span id="imageLabel"></span><a id="original" target="_blank" rel="noopener">打开原图</a><button id="close">关闭</button></div></dialog>
<script id="gallery-data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('gallery-data').textContent),rows=data.rows,$=id=>document.getElementById(id),labels={happy:'开心',sad:'难过',angry:'生气',funny:'搞笑',shock:'惊讶',cute:'卖萌'};
$('summary').textContent=`共 ${rows.length} 张 · 允许随机发送 ${rows.filter(r=>r.random_eligible===1&&['reaction','joke'].includes(r.meme_kind)).length} 张 · 导出时间 ${data.exported_at}`;
for(const t of [...new Set(rows.map(r=>r.topic))].sort()){const o=document.createElement('option');o.value=t;o.textContent=`${labels[t]||t} / ${t}`;$('topic').append(o)}
let page=0,filtered=[],size=60;
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function openImage(r){$('large').src=r.image;$('original').href=r.image;$('imageLabel').textContent=`#${r.id}`;$('viewer').showModal()}
function render(){const pages=Math.max(1,Math.ceil(filtered.length/size));page=Math.min(page,pages-1);$('grid').replaceChildren();for(const r of filtered.slice(page*size,(page+1)*size)){const card=el('article','card'),pic=el('div','picture'),img=el('img');img.src=r.image;img.alt=`#${r.id} ${r.description||'无描述'}`;img.loading='lazy';img.decoding='async';img.onclick=()=>openImage(r);img.onerror=()=>{pic.replaceChildren(el('span','meta','本地图片文件缺失'))};pic.append(img);card.append(pic);const body=el('div','body'),badges=el('div','badges');badges.append(el('strong','',`#${r.id}`),el('span','badge',r.meme_kind==='reaction'?'情绪反应图':r.meme_kind==='joke'?'网络梗图':'非梗图 / 未确认'),el('span','badge',`${labels[r.topic]||r.topic} / ${r.topic}`),el('span','badge '+(r.random_eligible===0?'off':r.random_eligible===null?'unknown':''),r.random_eligible===1?'允许随机发送':r.random_eligible===0?'禁止随机发送':'尚未标记'));body.append(badges,el('p','description',r.description||'（未保存描述）'),el('div','meta',`收录 ${r.created_at.replace('T',' ').slice(0,19)} · 已发送 ${r.used_count||0} 次`));const details=el('details'),summary=el('summary','','记忆短描述 / 文件标识');details.append(summary,el('p','description',r.memory_description||'（无短描述）'),el('div','meta',r.hash));body.append(el('p','description','分类依据：'+(r.classification_reason||'未保存')));body.append(el('div','meta','图中正文句数：'+(r.text_sentence_count??'未记录')));body.append(details);card.append(body);$('grid').append(card)}if(!filtered.length)$('grid').append(el('div','empty','没有符合条件的图片'));$('page').textContent=`${page+1} / ${pages}`;$('count').textContent=`找到 ${filtered.length} 张`;$('prev').disabled=page===0;$('next').disabled=page>=pages-1}
function apply(){const q=$('query').value.toLowerCase().trim();filtered=rows.filter(r=>(!q||`${r.id} ${r.hash} ${r.description||''} ${r.memory_description||''}`.toLowerCase().includes(q))&&(!$('topic').value||r.topic===$('topic').value)&&(!$('kind').value||(r.meme_kind||'unclassified')===$('kind').value)&&(!$('eligible').value||String(r.random_eligible)===$('eligible').value));filtered.sort($('sort').value==='old'?(a,b)=>a.id-b.id:$('sort').value==='used'?(a,b)=>(b.used_count||0)-(a.used_count||0)||b.id-a.id:(a,b)=>b.id-a.id);page=0;render()}
$('query').oninput=apply;for(const id of ['topic','kind','eligible','sort'])$(id).onchange=apply;$('prev').onclick=()=>{page--;render();window.scrollTo(0,0)};$('next').onclick=()=>{page++;render();window.scrollTo(0,0)};$('close').onclick=()=>$('viewer').close();$('viewer').onclick=e=>{if(e.target===$('viewer'))$('viewer').close()};apply();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Existing exported JSON; otherwise query Docker read-only')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.input:
        rows = json.loads(args.input.read_text(encoding='utf-8-sig'))
    else:
        query = "import sqlite3,json; c=sqlite3.connect('file:/app/data/memes.db?mode=ro',uri=True); c.row_factory=sqlite3.Row; print(json.dumps([dict(r) for r in c.execute('select id,hash,topic,description,memory_description,file_path,created_at,random_eligible,used_count,meme_kind,classification_reason,text_sentence_count from memes order by id desc')],ensure_ascii=False)); c.close()"
        result = subprocess.run(['docker', '--context', 'desktop-linux', 'exec', 'qq-llm-bot', 'python', '-c', query], capture_output=True, check=True, encoding='utf-8')
        rows = json.loads(result.stdout)
    missing = []
    for row in rows:
        matches = list((root / 'data/memes' / row['hash'][:2]).glob(row['hash'] + '.*'))
        if matches:
            row['image'] = matches[0].relative_to(root / 'data').as_posix()
        else:
            row['image'] = 'memes/missing-' + str(row['id'])
            missing.append(row['id'])
        row.pop('file_path', None)
    payload = json.dumps({'exported_at': datetime.now().isoformat(timespec='seconds'), 'rows': rows}, ensure_ascii=False).replace('<', chr(92) + 'u003c')
    output = root / 'data/meme-gallery.html'
    output.write_text(TEMPLATE.replace('__DATA__', payload), encoding='utf-8')
    print(json.dumps({'path': str(output), 'images': len(rows), 'missing_images': missing}, ensure_ascii=False))


if __name__ == '__main__':
    main()
