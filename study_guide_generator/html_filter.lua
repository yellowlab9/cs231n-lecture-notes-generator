function Meta(meta)
  meta['header-includes'] = meta['header-includes'] or pandoc.List()
  table.insert(meta['header-includes'], pandoc.RawBlock('latex', '\\usepackage{graphicx}'))
  table.insert(meta['header-includes'], pandoc.RawBlock('latex', '\\usepackage{float}'))
  return meta
end

local function resolve_img_path(src)
  return src:gsub("\\", "/")
end

function Image(el)
  el.src = resolve_img_path(el.src)
  el.attributes['width'] = '75%'
  return el
end

function RawBlock(el)
  if el.format == "html" then
    local src = el.text:match('src="([^"]+)"')
    local alt = el.text:match('alt="([^"]+)"') or ""
    local width_str = el.text:match('width="([^"]+)"') or "75%"
    if src then
      local width_num = width_str:gsub("%%", "")
      local width_frac = (tonumber(width_num) or 75) / 100
      local actual_src = resolve_img_path(src)
      local latex = string.format(
        "\\begin{figure}[H]\\centering\\includegraphics[width=%.2f\\linewidth]{%s}\\caption{%s}\\end{figure}",
        width_frac, actual_src, alt
      )
      return pandoc.RawBlock("latex", latex)
    end
  end
end

function RawInline(el)
  if el.format == "html" then
    local src = el.text:match('src="([^"]+)"')
    local alt = el.text:match('alt="([^"]+)"') or ""
    local width_str = el.text:match('width="([^"]+)"') or "75%"
    if src then
      local width_num = width_str:gsub("%%", "")
      local width_frac = (tonumber(width_num) or 75) / 100
      local actual_src = resolve_img_path(src)
      local latex = string.format(
        "\\begin{figure}[H]\\centering\\includegraphics[width=%.2f\\linewidth]{%s}\\caption{%s}\\end{figure}",
        width_frac, actual_src, alt
      )
      return pandoc.RawInline("latex", latex)
    end
  end
end
