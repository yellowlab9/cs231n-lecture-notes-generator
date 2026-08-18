function RawBlock(el)
  if el.format == "html" then
    local src = el.text:match('src="([^"]+)"')
    local alt = el.text:match('alt="([^"]+)"') or ""
    local width = el.text:match('width="([^"]+)"') or "75%"
    if src then
      local md = string.format('![%s](%s "%s"){ width=%s }', alt, src, alt, width)
      local doc = pandoc.read(md, "markdown+implicit_figures")
      if doc and doc.blocks and #doc.blocks > 0 then
        return doc.blocks[1]
      end
    end
  end
end

function RawInline(el)
  if el.format == "html" then
    local src = el.text:match('src="([^"]+)"')
    local alt = el.text:match('alt="([^"]+)"') or ""
    local width = el.text:match('width="([^"]+)"') or "75%"
    if src then
      local md = string.format('![%s](%s "%s"){ width=%s }', alt, src, alt, width)
      local doc = pandoc.read(md, "markdown+implicit_figures")
      local inlines = pandoc.utils.blocks_to_inlines(doc.blocks)
      if inlines and #inlines > 0 then
        return inlines[1]
      end
    end
  end
end
