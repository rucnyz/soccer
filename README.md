# 数据科学实践第一次项目

## 数据集准备

从[European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer) 下载`database.sqlite`置于`./data/`
目录下，之后使用`extract_sql.py`转为csv。

或者直接将所有csv文件放在`./data/`
目录下，包括

- country.csv
- league.csv
- match.csv
- player.csv
- player_attr.csv
- team.csv
- team_attr.csv

## 运行方法

- 得到所有统计建模的结果
    - 运行方式见下，共计需要92分钟左右。
    - 不对特征进行可视化，则将visual设置为0。

```shell
python main.py --seed 42 --visual 1 --metric f1
```

- 得到可视化结果
    - 运行`visualization.py`，需要30分钟左右