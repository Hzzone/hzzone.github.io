### 宜搜小说 API 分析

类似的有 [追书神器 API](https://github.com/xiadd/zhuishushenqi)，很久以前还挺好用的，但这应用收费章节比原版还贵，收费章节直接加密；后来直接连免费章节也不能用了，不能获取章节内容。

我做过的包括 [微信小程序](https://github.com/Hzzone-Archives/YourReader)，[Android](https://github.com/Hzzone/E-Book)，都是使用追书神器 API，现在失效了。

接下来介绍宜搜小说 API。

* 搜索

url: `http://api.easou.com/api/bookapp/search.m?word=红楼梦&type=0&sort_type=0&page_id=1&count=20&cid=eef_&version=002&os=ios&udid=76fbeddf8b73c8b0fc065e2296b767579d8fda38&appverion=1018&ch=bnf1349_10388_001
`

`get`，改变 word 参数。

response:
```json

{
    "type": 0,
    "subClass": "",
    "keyWord": "",
    "success": true,
    "errorlog": "",
    "sortType": 0,
    "all_book_items": [
        {
            "name": "红楼梦",
            "classes": "连载,其他",
            "desc": "《红楼梦》，作者曹雪芹，中国古代四大名著之一，章回体长篇小说......",
            "status": "连载",
            "gid": 17384298,
            "category": "其他",
            "chapterCount": 120,
            "cpId": "0",
            "charge": 0,
            "agentName": "",
            "ad": 0,
            "activityInfo": {
                "discount": 0,
                "price": 0,
                "originPrice": 0,
                "activityName": "",
                "activityType": -1,
                "ttl": 0
            },
            "activityType": -1,
            "nid": 19704996,
            "author": "官方好书推荐",
            "site": "lindiankanshu.com",
            "chargeGid": 0,
            "sourceId": "3035306",
            "imgUrl": "http://image.book.easou.com/i/default/cover.jpg",
            "lastChapterName": "第一百二十回 甄士隐详说太虚情 贾雨村归结红楼梦",
            "lastTime": 1509105932186,
            "subscribeCount": 180910,
            "tempFree": false,
            "siteCount": 0,
            "topicGroupId": 3515373,
            "topicNum": 2
        },
        ...
    ],
    "web_book_items": [],
    "guess_like_items": [],
    "check_word": "",
    "allTotal": 56,
    "parentClass": "",
    "guessLikeReferGids": "",
    "buttonToggle": "00",
    "webTotal": 0,
    "recforsoft": []
}
```

* 章节目录

url: `http://api.easou.com/api/bookapp/chapter_list.m?gid=17384298&nid=19704996&page_id=1&size=9999&cid=eef_&version=002&os=ios&udid=76fbeddf8b73c8b0fc065e2296b767579d8fda38&appverion=1018&ch=bnf1349_10388_001`

`get`，改变 `gid` 和 `nid`（由搜索结果获得）。

response:
```json
{
    "novelName": "",
    "totalCount": 120,
    "items": [
        {
            "time": 1509105916000,
            "sort": 1,
            "charge": 0,
            "wordCount": 0,
            "site": "lindiankanshu.com",
            "nid": 19704996,
            "chapter_name": "第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀",
            "gsort": 0,
            "curl": "http://www.lindiankanshu.com/93_93446/4869424.html",
            "paid": 0,
            "ctype": "文",
            "accelerate": 0
        },
        ...
    ],
    "success": true,
    "errorlog": ""
}
```

* 章节内容
url: `http://api.easou.com/api/bookapp/batch_chapter.m?cid=eef_&version=002&os=ios&udid=76fbeddf8b73c8b0fc065e2296b767579d8fda38&appverion=1018&ch=bnf1349_10388_001`

`post`，`header：content-type: application/x-www-form-urlencoded`，提交以下参数:
```
{
    "gid": "17384298",
    "nid": "19704996",
    "gsort": 0,
    "chapter_name": "第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀",
    "sequence": 0,
    "sort": 120
}
```

`gid`、`nid`、`sort`、`chapter_name` 需要改变。

response:
```json
{
    "items": [
        {
            "content": "　　话说宝钗听秋纹说袭人不好，连忙进去瞧看。巧姐儿同平儿也随着走到袭人炕前，只见袭人心痛难禁，一时气厥。宝钗等用开水灌了过来，仍旧扶她睡下，一面传请大夫。巧姐儿问宝钗道：\"袭人姐姐怎么病到这个样？\"宝钗道：\"大前儿晚上，哭伤了心了，一时发晕栽倒了。太太叫人扶她回来，她就睡倒了。因外头有事，没有请大夫瞧她，所以致此。\"
            ...
            ",
            "time": 1509105932000,
            "flag": 0,
            "gid": 17384298,
            "sort": 120,
            "charge": 0,
            "sequence": 0,
            "nid": 19704996,
            "site": "",
            "chapter_name": "第一百二十回 甄士隐详说太虚情 贾雨村归结红楼梦",
            "tempFree": false,
            "curl": "http://www.lindiankanshu.com/93_93446/4869543.html",
            "paid": 0,
            "ctype": "文",
            "accelerate": 0,
            "success": true,
            "errorlog": ""
        }
    ],
    "success": true,
    "errorlog": ""
}
```

