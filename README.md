# 数据科学实践第一次项目

## 数据集准备

将所有csv文件放在`./data/`目录下，包括

- country.csv
- league.csv
- match.csv
- player.csv
- player_attr.csv
- team.csv
- team_attr.csv

## 运行方法

得到所有结果

```shell
python main.py --seed 42 --visual 1 --metric f1
```

不对特征进行可视化，则将visual设置为0