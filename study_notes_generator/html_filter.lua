function Meta(meta)
  meta['header-includes'] = meta['header-includes'] or pandoc.List()
  table.insert(meta['header-includes'], pandoc.RawBlock('latex', '\\usepackage{graphicx}'))
  table.insert(meta['header-includes'], pandoc.RawBlock('latex', '\\usepackage{float}'))
  return meta
end

local function resolve_img_path(src)
  return src:gsub("\\", "/")
end

-- Filter out web action badges (Colab, Shields.io, GitHub badges) from PDF
function Link(el)
  if el.target:find("colab.research.google.com") or el.target:find("shields.io") or el.target:find("raw.githubusercontent.com") or (el.target:find("releases/download") and el.target:find("%.pdf")) then
    for _, child in ipairs(el.content) do
      if child.t == "Image" and (child.src:find("colab%-badge") or child.src:find("shields%.io")) then
        return {}
      end
    end
  end
  return el
end

function Image(el)
  if el.src:find("colab%-badge") or el.src:find("shields%.io") then
    return {}
  end
  el.src = resolve_img_path(el.src)
  el.attributes['width'] = '75%'
  return el
end

function Para(el)
  local has_real_content = false
  for _, item in ipairs(el.content) do
    if item.t == "Str" and item.text:match("%S") and item.text ~= "&nbsp;" and item.text ~= " " then
      has_real_content = true
      break
    elseif item.t == "Image" or item.t == "Link" or item.t == "Math" or item.t == "RawInline" then
      has_real_content = true
      break
    end
  end
  if not has_real_content then
    return {}
  end
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
