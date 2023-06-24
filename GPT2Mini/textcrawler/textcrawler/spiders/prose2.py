import scrapy
import re
import os


class Prose2Spider(scrapy.Spider):
    name = "prose2"
    headers = {
        "User-Agent": "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; "
                      "Tablet PC 2.0; wbx 1.0.0; wbxapp 1.0.0; Zoom 3.6.0) "
    }

    base_urls = [
        {"base_url": "http://read.banbijiang.com/sanwentiandi/xiangtuqingjie/", "index": "list_33_", "pages": 72},
        {"base_url": "http://read.banbijiang.com/sanwentiandi/youqingwenzhang/", "index": "list_141_", "pages": 68},
        {"base_url": "http://read.banbijiang.com/sanwentiandi/qinqingwenzhang/", "index": "list_118_", "pages": 138},
        {"base_url": "http://read.banbijiang.com/sanwentiandi/sundays/", "index": "list_80_", "pages": 65},
        {"base_url": "http://read.banbijiang.com/sanwentiandi/ganchuxinling/", "index": "list_52_", "pages": 359},
        {"base_url": "http://read.banbijiang.com/sanwentiandi/aiqingwenzhang/", "index": "list_144_", "pages": 148},
        {"base_url": "http://read.banbijiang.com/sanwentiandi/zheliwenzhang/", "index": "list_123_", "pages": 59},
        {"base_url": "http://read.banbijiang.com/sanwentiandi/suiyueruge/", "index": "list_100_", "pages": 206},
        {"base_url": "http://read.banbijiang.com/sanwentiandi/tongnianjiyi/", "index": "list_60_", "pages": 35},
        {"base_url": "http://read.banbijiang.com/sanwentiandi/xiejingwenzhang/", "index": "list_170_", "pages": 44},
    ]

    custom_settings = {
        'ITEM_PIPELINES': {'textcrawler.pipelines.Prose2crawlerPipeline': 300}
    }

    def start_requests(self):
        for item in self.base_urls:
            base, index, pages = item['base_url'], item['index'], item['pages']

            for i in range(1, pages + 1):
                sub_index = '' if i == 1 else f'{index}{i}.html'
                url = os.path.join(base, sub_index)
                try:
                    yield scrapy.Request(url=url, callback=self.parse, headers=self.headers)
                except:
                    continue

    def parse(self, response):
        prose_urls = response.xpath('//ul[@class="e2"]/li/a/@href').extract()
        for href in prose_urls:
            try:
                yield scrapy.Request(url=href, callback=self.parse_page, headers=self.headers)
            except:
                continue

    def parse_page(self, response):
        text = response.xpath('//div[@class="content"]//p/text()').extract()
        merge_text = []
        for piece in text:
            piece = piece.strip()
            if len(piece) > 0:
                merge_text.append(piece)

        clean_text = '\n'.join(merge_text)

        # Any text which length less than 200 will be filtered out
        if len(clean_text) < 200:
            pass
        else:
            yield {'text': clean_text}
