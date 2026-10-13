"""Authored synthetic dialogue fixtures; no real chat records."""


def cases():
    """Exactly 100 chronological messages in each case, ending in other speech."""

    def history():
        return [
            {"role": "self" if i % 2 else "other", "text": f"今天第{i + 1}件日常小事是整理桌面，没有新的约定或改变。"}
            for i in range(100)
        ]

    rows = history()
    rows[0] = {"role": "other", "text": "我的论文绪论要重写研究问题，这一章折腾我两天了。"}
    rows[1] = {"role": "self", "text": "你改完论文绪论可以发给我，我帮你看看研究问题。"}
    rows[99] = {"role": "other", "text": "论文绪论终于改完了，今天这块硬骨头算被我啃下来了。"}
    yield "concrete_progress", rows, ("论文", "绪论", "研究问题")

    rows = history()
    rows[0] = {"role": "other", "text": "我准备这场实习面试一整周了，结果周五公布。"}
    rows[1] = {"role": "self", "text": "这周你练了好几次，我知道你为实习面试花了很多时间。"}
    rows[99] = {"role": "other", "text": "面试没过，明明准备那么久，还是觉得自己很没用。"}
    yield "low_mood", rows, ("面试", "实习", "准备", "没过")

    rows = history()
    rows[0] = {"role": "other", "text": "周六聚餐先约晚上七点，我不吃辣。"}
    rows[1] = {"role": "self", "text": "好，我来找餐厅，菜里不放辣椒。"}
    rows[74] = {"role": "other", "text": "更正一下，聚餐改成晚上八点，七点我赶不到。"}
    rows[75] = {"role": "self", "text": "记好了，周六晚上八点，餐厅还是我来选。"}
    rows[99] = {"role": "other", "text": "那就按刚才改的时间，餐厅你来定吧。"}
    yield "later_time_correction", rows, ("餐厅", "八点", "8点", "20:00", "不辣", "清淡")

    rows = history()
    rows[0] = {"role": "self", "text": "周五我和你去球馆打羽毛球，上次我迟到五分钟。"}
    rows[1] = {"role": "other", "text": "好，我先拿拍子，你这次别让我等到变成球馆雕像。"}
    rows[99] = {"role": "other", "text": "你再迟到我就宣布你是羽毛球界的拖延冠军，罚跑两圈！"}
    yield "light_teasing", rows, ("羽毛球", "球馆", "球", "迟到", "罚跑", "冠军")
