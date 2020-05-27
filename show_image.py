import cv2

IMAGE_PATH = 'map.PNG'


def set_coloring_for_image(coloring):
    colors_num = {}
    for v, color in coloring[0].items():
        if color == 'G':
            colors_num[v.ID] = [152, 251, 152]
        if color == 'R':
            colors_num[v.ID] = [204, 255, 255]
        if color == 'B':
            colors_num[v.ID] = [245, 147, 101]
        if color == 'L':
            colors_num[v.ID] = [147, 112, 219]
    return colors_num

img = cv2.imread(IMAGE_PATH)

def set_data(colors_i):
    part1 = (41, 58)
    part2 = (90, 56, 175)
    part3 = (118, 57, 108)
    part5 = (195, 130)
    part6 = (42, 212, 290)
    part7 = (89, 212)
    part9 = (156, 174)
    part10 = (201, 273, 322, 110)
    part11 = (218, 15, 8)
    part12 = (218, 160, 8, 373, 265)

    img[part1[0]:85, part1[1]:208] = colors_i[1]  # all

    img[part2[0]:113, part2[1]:208] = colors_i[2]  # up
    img[part2[0]:153, part2[2]:208] = colors_i[2]  # down-side

    img[part3[0]:163, part3[1]:170] = colors_i[3]  # main
    img[163:174, part3[2]:170] = colors_i[3]
    img[174:192, 156:170] = colors_i[3]  # small squre

    img[167:213, 58:103] = colors_i[4]  # main
    img[178:213, 103:127] = colors_i[4]  # down
    img[178:192, 103:154] = colors_i[4]  # side

    img[part5[0]:214, part5[1]:181] = colors_i[5]

    img[part6[0]:85, part6[1]:368] = colors_i[6]  # main
    img[part6[0]:105, part6[2]:368] = colors_i[6]  # down-side

    img[part7[0]:125, part7[1]:285] = colors_i[7]  # all

    img[109:197, 288:318] = colors_i[8]  # left
    img[129:183, 250:318] = colors_i[8]
    img[175:197, 273:290] = colors_i[8]  # small squre
    img[129:153, 211:318] = colors_i[8]

    img[part9[0]:191, part9[1]:247] = colors_i[9]  # up
    img[185:214, 185:270] = colors_i[9]  # down

    img[part10[3]:212, part10[2]:368] = colors_i[10]  # middel
    img[part10[0]:213, part10[1]:368] = colors_i[10]  # down

    img[part11[0]:250, part11[1]:155] = colors_i[11]  # down
    img[part11[1]:240, part11[1]:54] = colors_i[11]  # side
    img[part11[2]:37, part11[1]:260] = colors_i[11]  # up

    img[part12[2]:37, part12[4]:420] = colors_i[12]  # up
    img[part12[2]:250, part12[3]:420] = colors_i[12]  # side
    img[part12[0]:250, part12[1]:420] = colors_i[12]  # down

def image(color_i):
    set_data(color_i)
    cv2.imshow("img", img)
    key = cv2.waitKey()
    cv2.destroyAllWindows()