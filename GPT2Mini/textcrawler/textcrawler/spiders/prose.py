import scrapy
import re
import os


class ProseSpider(scrapy.Spider):
    name = "prose"
    headers = {
        "User-Agent": "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; "
                      "Tablet PC 2.0; wbx 1.0.0; wbxapp 1.0.0; Zoom 3.6.0) "
    }

    base_urls = [
        {"base_url": "https://www.xinsanwen.cn/sanwen/xushi", "pages": 1794},
        {"base_url": "https://www.xinsanwen.cn/sanwen/shuqing", "pages": 4319}
    ]

    custom_settings = {
        'ITEM_PIPELINES': {'textcrawler.pipelines.ProsecrawlerPipeline': 300}
    }

    def start_requests(self):
        for item in self.base_urls:
            base, pages = item['base_url'], item['pages']

            for i in range(1, pages + 1):
                index = 'index.html' if i == 1 else f'index_{i}.html'
                url = os.path.join(base, index)
                try:
                    yield scrapy.Request(url=url, callback=self.parse, headers=self.headers)
                except:
                    continue

    def parse(self, response):
        prose_urls = response.xpath('//div[@class="blogs-list"]/ul/li/h2/a/@href').extract()
        for href in prose_urls:
            try:
                yield scrapy.Request(url=href, callback=self.parse_page, headers=self.headers)
            except:
                continue

    def parse_page(self, response):
        text = response.xpath('//div[@class="newstext"]/text()').extract()
        merge_text = []
        for piece in text:
            piece = piece.strip()
            if len(piece) > 0:
                merge_text.append(piece)

        clean_text = self.clean('\n'.join(merge_text))

        # Any text which length less than 200 will be filtered out
        if len(clean_text) < 200:
            pass
        else:
            yield {'text': clean_text}

    @staticmethod
    def clean(text):
        clean_text = re.sub(r'\n*\u3000+', '\n', text)  # \n加多个空格
        clean_text = re.sub(' +', '', clean_text)  # 空格消除
        clean_text = re.sub(r'\n*\?\?', '\n', clean_text)  # \n??或者??
        clean_text = re.sub('\n+', '\n', clean_text)  # 多个\n
        return clean_text.strip()
