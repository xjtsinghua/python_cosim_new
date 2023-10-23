


def _init():
    global tag
    tag=0
def tag_setValue(value):
    global tag
    tag=value
def tag_getValue():
    try:
        return tag
    except:
        print('无值')


