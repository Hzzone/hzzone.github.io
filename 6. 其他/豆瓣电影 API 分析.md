### 豆瓣电影 API

1. 电影正在热映

url: http://api.douban.com/v2/movie/nowplaying?apikey=0df993c66c0c636e29ecbb5344252a4a

2. 电影信息

url: https://api.douban.com/v2/movie/subject/:id

para:
* id: 电影 id

example: https://api.douban.com/v2/movie/subject/3168101

3. 电影剧照

url: http://api.douban.com/v2/movie/subject/:id/photos?apikey=0df993c66c0c636e29ecbb5344252a4a

para:
* id: 电影 id

example: http://api.douban.com/v2/movie/subject/3168101/photos?apikey=0df993c66c0c636e29ecbb5344252a4a

4. 明星信息

url: http://api.douban.com/v2/movie/celebrity/:id?apikey=0df993c66c0c636e29ecbb5344252a4a

para:
* id: 明星 id

example: http://api.douban.com/v2/movie/celebrity/1019015?apikey=0df993c66c0c636e29ecbb5344252a4a

5. 明星剧照

url: http://api.douban.com/v2/movie/celebrity/:id/photos?apikey=0df993c66c0c636e29ecbb5344252a4a

para:
* id: 明星 id

example: http://api.douban.com/v2/movie/celebrity/1019015/photos?apikey=0df993c66c0c636e29ecbb5344252a4a
             
6. 即将上映

url: http://api.douban.com/v2/movie/coming?apikey=0df993c66c0c636e29ecbb5344252a4a

8. TOP250

url: http://api.douban.com/v2/movie/top250?apikey=0df993c66c0c636e29ecbb5344252a4a

para:
* start: 数据的开始项
* count: 单页条数

example: 获取电影 Top 250 第一页 10 条数据: https://api.douban.com/v2/movie/top250?start=0&count=10

9. 电影本周口碑榜

url: http://api.douban.com/v2/movie/weekly?apikey=0df993c66c0c636e29ecbb5344252a4a

10. 北美票房榜

url: http://api.douban.com/v2/movie/us_box?apikey=0df993c66c0c636e29ecbb5344252a4a

11. 新片榜

url: http://api.douban.com/v2/movie/new_movies?apikey=0df993c66c0c636e29ecbb5344252a4a

12. 搜索

url: https://api.douban.com/v2/movie/search?q=keyword?tag=keyword

para:
* start: 数据的开始项
* count: 单页条数
* q: 电影名称
* tag: 电影标签

example: 
* 根据电影吗名称搜索: https://api.douban.com/v2/movie/search?q=色戒&start=0&count=10
* 根据电影标签搜索: https://api.douban.com/v2/movie/search?tag=华语&start=0&count=10

13. 获取电影院正在热映的电影

url: https://api.douban.com/v2/movie/in_theaters

para:
* start: 数据的开始项
* count: 单页条数
* city: 城市

example: 
* https://api.douban.com/v2/movie/in_theaters?city=成都&start=0&count=10

# 参考[豆瓣电影API](https://developers.douban.com/wiki/?title=movie_v2 "豆瓣电影API")#