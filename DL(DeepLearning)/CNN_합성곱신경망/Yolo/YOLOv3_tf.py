def detector():
  ima_raw = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)

  img = tf.expand_dims(ima_raw, 0)  #디멘젼 형태로 변환
  img = transform_images(img, FLAGS.size)

  t1 = time.time()
  boxs, scores, classes, nums = yolo(img)
  t2 = time.time() 
  #t2-t1을 빼면 실제 yolo가 실행된 시간
  print('time: {}'.format(t2-t1))

  for i in range(num[0]):
    .print('{}, {}, {}'.format(class_names[int(classes[0][i])],
                               np.array(scores[0][i]),
                               np.array(boxes[0][i])))
  #BGR: 그림그리기, draw_output: 표시
  img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
  img = draw_output(img, (boxes, scores, classes, num), class_names)

  return img