function Meta(meta)
  meta['header-includes'] = meta['header-includes'] or pandoc.List()
  table.insert(meta['header-includes'], pandoc.RawBlock('latex', '\\usepackage{graphicx}'))
  return meta
end

local function resolve_img_path(src)
  -- If src exists as is, use it
  local f = io.open(src, "r")
  if f then
    f:close()
    return src
  end
  -- If not found, check inside lectures/<lecture_name>/
  local lec = src:match("(lecture_%d+)")
  if lec then
    local candidate = "lectures/" .. lec .. "/" .. src
    local f2 = io.open(candidate, "r")
    if f2 then
      f2:close()
      return candidate
    end
  end
  return src
end

function RawBlock(el)
  if el.format == "html" then
    local src = el.text:match('src="([^"]+)"')
    local alt = el.text:match('alt="([^"]+)"') or ""
    local width_str = el.text:match('width="([^"]+)"') or "75%"
    if src then
      local width_num = width_str:gsub("%%", "")
      local width_frac = (tonumber(width_num) or 75) / 100
      local caption_str = alt ~= "" and alt or "Slide"
      local actual_src = resolve_img_path(src)
      local latex = string.format(
        "{\\centering\\noindent\\includegraphics[width=%.2f\\linewidth]{%s}\\par\\vspace{3pt}{\\small\\textit{%s}}\\par\\vspace{10pt}}",
        width_frac, actual_src, caption_str
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
      local caption_str = alt ~= "" and alt or "Slide"
      local actual_src = resolve_img_path(src)
      local latex = string.format(
        "{\\centering\\noindent\\includegraphics[width=%.2f\\linewidth]{%s}\\par\\vspace{3pt}{\\small\\textit{%s}}\\par\\vspace{10pt}}",
        width_frac, actual_src, caption_str
      )
      return pandoc.RawInline("latex", latex)
    end
  end
end
