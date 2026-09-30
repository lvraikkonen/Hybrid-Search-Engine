#!/usr/bin/env python3
"""Build the S5 HTTP app shell; render() remains the S2 document sanitizer."""
from pathlib import Path
import json,hashlib,re
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup, NavigableString
import bleach
ROOT=Path(__file__).resolve().parent
MD=MarkdownIt('commonmark',{'html':True,'breaks':False,'linkify':False}).enable(['table','strikethrough'])
# Presentation is separately derived; source text is always preserved byte-for-byte as decoded.
TAGS=['p','br','hr','strong','em','b','i','u','s','del','blockquote','pre','code','ul','ol','li','h1','h2','h3','h4','h5','h6','table','thead','tbody','tfoot','tr','th','td','caption','colgroup','col','a','img','div','span','details','summary','sub','sup','kbd','abbr','dl','dt','dd']
ATTRS={'a':['href','title'],'img':['src','alt','title'],'th':['colspan','rowspan','scope'],'td':['colspan','rowspan'],'ol':['start','reversed'],'li':['value'],'code':['class'],'abbr':['title']}

def render(raw, prefix):
    """Sanitize derived HTML; remove scripts/styles; never load external images."""
    parsed=BeautifulSoup(MD.render(raw or ''),'html.parser')
    decorations=len(parsed.select('svg'))
    for el in parsed.select('script,style,iframe,object,embed,svg,math,form,button,input,textarea,select,video,audio,source,link,meta,base'):
        el.decompose()
    # Source HTML sometimes contains literal Markdown inside table cells.
    # Render text-node Markdown in td/th only; leave tables and existing links intact.
    for textnode in list(parsed.find_all(string=True)):
        if textnode.find_parent(['td','th']) and not textnode.find_parent(['a','pre','code']):
            s=str(textnode)
            if re.search(r'\[[^\]]+\]\([^\n]+\)|\*\*[^*]+\*\*',s):
                part=BeautifulSoup(MD.renderInline(s),'html.parser')
                for ch in list(part.contents):textnode.insert_before(ch)
                textnode.extract()
    clean=bleach.clean(str(parsed),tags=TAGS,attributes=ATTRS,protocols=['http','https','mailto'],strip=True,strip_comments=True)
    soup=BeautifulSoup(clean,'html.parser')
    search_text=soup.get_text(' ',strip=True)
    assets=[]
    for img in list(soup.find_all('img')):
        src=img.get('src','');alt=img.get('alt') or '文档插图'
        assets.append({'kind':'image','path':src,'label':alt})
        box=soup.new_tag('div',attrs={'class':'asset-placeholder','role':'note'})
        sym=soup.new_tag('span',attrs={'class':'asset-symbol','aria-hidden':'true'});sym.string='▧';box.append(sym)
        body=soup.new_tag('div');strong=soup.new_tag('strong');strong.string=alt;body.append(strong)
        info=soup.new_tag('span');info.string='图片未随 SQL 提供 · 保留原始引用';body.append(info)
        code=soup.new_tag('code');code.string=src;body.append(code);box.append(body);img.replace_with(box)
    for a in soup.find_all('a'):
        href=a.get('href','')
        # Data attributes below are application-generated, never accepted from source HTML.
        if href.startswith('#'):
            a['data-anchor']=href[1:]; a['href']='#'
        elif re.match(r'^(https?://|mailto:)',href,re.I):
            a['target']='_blank';a['rel']='noopener noreferrer';a['referrerpolicy']='no-referrer'
            a['class']='external-link'
        else:
            a['data-local-link']=href; a['href']='#';a['class']='local-link'
            if not (href.startswith('/zh-cn') or href.startswith('/en-us')):
                assets.append({'kind':'attachment','path':href,'label':a.get_text(' ',strip=True)})
    # $learnfiles...$ placeholders outside Markdown links also remain visible, not fetched.
    for textnode in list(soup.find_all(string=True)):
        if textnode.find_parent(['a','code','pre']):continue
        s=str(textnode)
        if re.search(r'\$(?:learnfiles|images)[^$\r\n]+\$',s,re.I):
            matches=list(re.finditer(r'\$(?:learnfiles|images)[^$\r\n]+\$',s,re.I));pos=0
            for m in matches:
                if m.start()>pos:textnode.insert_before(NavigableString(s[pos:m.start()]))
                path=m.group(0);a=soup.new_tag('a',attrs={'href':'#','data-local-link':path,'class':'local-link'});a.string=path
                textnode.insert_before(a);assets.append({'kind':'attachment','path':path,'label':path});pos=m.end()
            if pos<len(s):textnode.insert_before(NavigableString(s[pos:]))
            textnode.extract()
    toc=[]
    for i,h in enumerate(soup.find_all(re.compile(r'^h[1-6]$'))):
        hid=f'{prefix}-h{i}';h['id']=hid;toc.append({'id':hid,'level':int(h.name[1:]),'text':h.get_text(' ',strip=True)})
    for t in list(soup.find_all('table')):
        wrap=soup.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'文档表格，可横向滚动'});t.wrap(wrap)
    text=soup.get_text(' ',strip=True)
    return {'html':str(soup),'text':search_text,'toc':toc,'assets':assets,'omitted_decorations':decorations}

def main():
    destination = ROOT / 'web'
    if destination.is_symlink():
        raise ValueError('Static directory must not be a symlink')
    destination.mkdir(exist_ok=True)
    assets = {}
    for source, name in [('template.html', 'index.html'), ('style.css', 'style.css'), ('app.js', 'app.js')]:
        content = (ROOT / 'src' / source).read_bytes()
        target = destination / name
        if target.is_symlink():
            raise ValueError('Static assets must not be symlinks')
        target.write_bytes(content)
        assets[name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    report = {'mode': 'same_origin_api_shell', 'assets': assets, 'embedded_documents': 0,
              'legacy_html_and_portal_data': 'preserved_offline_not_served'}
    (ROOT / 'reports/S5_BUILD.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
