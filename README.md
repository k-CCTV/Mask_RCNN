| 주제 | (강 주변) 위험 사고 검출을 위한 지능형 CCTV 기술 |
| --- | --- |
| 기간 | 2022.03.01 ~ 2022.11.17 |
| 인원 | 3명 |
| 개발 언어 | - Python
- Java script |
| 간단 소개 | 강 주변 사람에게 일어나는 위험 사고 영상에서 위험 단계를 출력 |
| 깃허브 링크 | https://github.com/k-CCTV |

## 목차

1. 프로젝트 개요
2. 사용 기술 및 개발 환경
3. 내용
4. 프로젝트 후기

## 1. 프로젝트 개요

- 소개

강 주변에서 물 빠짐 등의 사람의 위험사고를 감지하는 시스템

- 내 역할
- 시스템 통합, 내부로직 구현, 데이터 수집 및 학습
    
    시스템 통합 :
    
    - [ ]  Django와 강 경계 검출 모듈, 사람 검출 모듈, 위험 단계 계산 모듈 한번에 수행할 수 있도록 통합
    
    내부로직 구현 :
    
    - [ ]  강 경계와 육지 경계 구분하여 영상에 표시 후 저장
    - [ ]  경계 검출 영상을 토대로 사람 검출 영상 저장
    - [ ]  강과 사람 사이의 거리를 계산하여 위험단계 저장
    - [ ]  Django 서버에 검출 결과와 영상 전송, DB 갱신
    
    데이터 수집 및 학습 :
    
    - [ ]  학습을 위한 강 이미지, 영상을 강 경계와 육지 직접 라벨링
    - [ ]  직접 촬영, COCO Dataset, Youtube에서 이미지, 영상 데이터 수집
    

- 핵심 기능
1. Semantic Segmentation인 Mask RCNN을 이용한 물과 육지의 경계 검출
2. Object Detection인 Yolov5를 이용한 강 주변 사람의 행동 검출
3. 분할 영역과 검출된 사람 정보를 이용하여 이상 상황 판단
4. Web Page에 영상을 올려서 객체 검출 결과를 확인

## 2. 사용 기술 및 개발 환경

- 사용 기술

| Language | Python, Java Script |
| --- | --- |
| Backend | Django |
| Frontend | Django |
| Semantic Segmantation | Mask R-CNN (Tensorflow) |
| Object Detction | YOLOv5 (Pytorch) |
| DB | MySQL |
| Dataset | COCO dataset, yaml dataset, Youtube, 직접 촬양 |

- 개발 환경

| 이름 | 버전 | 목적 |
| --- | --- | --- |
| Python | 3.7 | 파이썬 코드를 실행하기 위한 패키지 |
| Django | 3.2.15 | Python 웹 인터페이스 서버 구동 |
| Anaconda | 4.12.0 | 로컬 환경에서의 딥러닝을 실행하기 위한 패키지 |
| MySQL | 8.0.20 | DB 저장 및 검색 |
| Node JS | 16.17.0 | 웹 인터페이스를 구축하기 위한 패키지 |

## 3. 내용

- 기능 개요
1. 강 육지 검출 
- 강 경계를 구분하기 위해 강 영역을 무작위 색깔로 채우고 ROI를 추출한다

1. 강에서 발생할 수 있는 사람의 위험사고 검출
- 사람 객체를 검출하고 Bbox를 추출하여 사람에게 박스를 그리고 강에 빠져있는지 검출한다

1. 문제 상황 발생 시 감시 단계 설정한다. 
- 강의 ROI와 사람의 Bbox의 좌표를 계산하여 서로 겹친 정도를 확인한다
- 감시 단계를 정상, 경고, 위험의 3단계로 나눈다.
- 사람이 강에 빠져있는 것을 검출하면 바로 위험단계로 설정한다.

1. Web Page에 영상을 올려서 객체 검출 결과를 확인
- 영상을 Web Page 업로드하여 위 과정들을 걸친 후, 검출 결과와 검출 영상을 확인한다.

### 동작 영상

![result_2](https://github.com/user-attachments/assets/8bc9c2f7-9aeb-4279-a8db-0c90b4e885a3)


- [ ]  강 영역 탐지 코드 일부

```python
#영상 인자 전송
capture = cv2.VideoCapture(video_url)
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
#영상 코덱 설정
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter(video_arg, codec, 24.0, size)
get_roi = False
river_mask_roi = None
count_num = 0

#프레임 별로 강 영역 검출 후 영상에 기록
while (capture.isOpened()):
    ret, frame = capture.read()
    if ret:
		    #영상 프레임이 존재할 시 영역 탐지 시작
        results = model.detect([frame], verbose=0)
        # verbose =1 이면 출력 0이면 출력 x
        ax = get_ax(1)
        r = results[0]
        frame = visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                            dataset.class_names, r['scores'], ax=ax,
                                            title="Predictions")

			  # 강 영역 좌표 저장
        save_roi = r['rois'].astype(int)
        if len(save_roi) > 0:
            save_rois = np.concatenate(save_roi).tolist()
            with open(MASK_ROOT / 'roi.txt', 'a') as text_f:
                text_f.write(str(save_rois))
                
        #강 여러개일때 고려
        with open(MASK_ROOT / 'roi.txt', 'a') as text_f:
            text_f.write("\n")
            
        #검출 후 영상에 작성
        output.write(frame)
        ax = plt.clf()
        count_num = count_num + 1
        print(str(count_num) + "  " + str(save_rois))
        
        #영상 검출 멈출 시
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    else: # 마지막 프레임이 끝났을 경우
        print("Done")
        break
```

- [ ]  내 프로젝트에 맞춰 Mask-RCNN 오픈소스 변형 코드 일부

```python
   	masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        
        #바운딩 박스 꼭지점 좌표 4개 따로 저장
        y1, x1, y2, x2 = boxes[i]
        #박스 있을 시 네모 모양 적용
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # 검출된 영역 이미지에 적용
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    # 적용된 이미지 및 강 영역 좌표 반환
    return masked_image.astype(np.uint8)
```

- [ ]  위험 단계 계산 코드 일부

```python
for temp1, temp2 in zip(yolo_data, roi_data):
	  #강 영역 검출된 좌표 저장된 텍스트 파일에서 roi 추출
    roi_num = re.findall("\d+", temp2)
    i = 2
    cnt += 1
    check_status = True
    #사람 검출된 Bbox 좌표 저장된 텍스트 파일에서 좌표 추출
    yolo_num = re.findall("\d+", temp1)
    
    # 사람이 강에 빠진 장면 검출 시 위험 횟수 + 1
    if 'personINthewater' in temp1:
        danger_cnt = danger_cnt + 1
    # 사람만 검출 시
    else:
	      # 강 영역 검출 시
        if len(roi_num) > 1:
            river_x1 = int(roi_num[1])
            river_y1 = int(roi_num[0])
            river_x2 = int(roi_num[3])
            river_y2 = int(roi_num[2])
            while len(yolo_num) > (i - 2):
                person_x1 = int(yolo_num[0 + i])
                person_y1 = int(yolo_num[1 + i])
                person_x2 = int(yolo_num[2 + i])
                person_y2 = int(yolo_num[3 + i])
                #사람이 두사람 이상 일때
                if len(yolo_num) > (i - 2):
                    i = i + 6
                else:
                    break
                # 강 영역 안에 사람 좌표가 다 있을 시
                if (person_x1 >= river_x1) and (person_y1 >= river_y1) and (person_x2 <= river_x2) and (person_y2 <= river_y2):
                    danger_cnt += 1
                    break
                else:
	                  # 강 영역에 사람 좌표가 좌측에서 걸쳐 있을 때
                    if (person_x1 >= river_x1) and (person_x1 <= river_x2):
                        if (person_y1 >= river_y1) and (person_y1 <= river_y2):
                            warn_cnt += 1
                            box_num = cal_box_percent(river_x1, river_x2, river_y1, river_y2, person_x1, person_x2, person_y1, person_y2)
                            box_area_num = box_area(person_x1, person_x2, person_y1, person_y2)
                            box_percent = float(box_num) / float(box_area_num)
                            box_percent_sum = box_percent_sum + box_percent
                            print("1 = " + str(box_num) + " " + str(box_area_num) + " " + str(box_percent) + " " + str(box_percent_sum))
                            break
                    # 강 영역에 사람 좌표가 우측에서 걸쳐 있을 때
                    elif (person_x2 >= river_x1) and (person_x2 <= river_x2):
                        if (person_y2 >= river_y1) and (person_y2 <= river_y2):
                            warn_cnt += 1
                            box_num = cal_box_percent(river_x1, river_x2, river_y1, river_y2, person_x1, person_x2, person_y1, person_y2)
                            box_area_num = box_area(person_x1, person_x2, person_y1, person_y2)
                            box_percent = float(box_num) / float(box_area_num)
                            box_percent_sum = box_percent_sum + box_percent
                            print("2 = " + str(box_num) + " " + str(box_area_num) + " " + str(box_percent) + " " + str(box_percent_sum))
                            break

#평균 퍼센트로 계산하여 위험단계 측정
box_percent_avg = box_percent_sum / warn_cnt * 100
```

- [ ]  검출 완료 시 Django 서버에 전송하는 코드 일부

```python

    print("검출경로 : " + detected_file_root)
    if not os.path.exists(media_root):
        os.makedirs(media_root)
        shutil.copy2(detected_file, detected_file_root)
        print("디렉토리 및 파일 이동 성공")
    else:
        shutil.copy2(detected_file, detected_file_root)
        print("파일 이동 성공")

    #경고, 위험, 수치 텍스트 파일 열람 후 저장
    warn_cnt = re.findall("\d+", status_result[0])
    danger_cnt = re.findall("\d+", status_result[1])
    result_percent = re.findall("\d+\.\d+", status_result[2])

    post_list = Board.objects.filter(id=db_id)
    
    # 위험 수치가 24 프레임(1초) 이상일 때 위험 상태로 DB 갱신
    if int(danger_cnt[0]) > 24:
        post_list.update(status=3, detect_files="detect/" + video, detact_result=100.0)
    # 경고 수치가 24 프레임(1초) 이상일 때 경고 상태로 DB 갱신
    elif int(warn_cnt[0]) > 24:
        post_list.update(status=2, detect_files="detect/" + video, detact_result=str(result_percent)[2:12])
    # 그 외는 안전 상태로 DB 갱신       
    else:
        post_list.update(status=1, detect_files="detect/" + video)
```

## 4. 프로젝트 후기

- **어려웠던 문제 및 해결**
- [ ]  라벨링된 데이터 학습시 너무 오랜 시간 소요

| **원인** | 라벨링된 데이터 학습시 GPU를 인식하지 못하고 CPU를 사용 |
| --- | --- |
| **해결** | - GPU코어 번호 지정 및 Tensorflow 1.13.1 버전을 Tensorflow-gpu 1.131 버전으로 바꿈으로서 
   GPU를 통한 학습
- CPU : Epoch 1회당 12시간이 넘게 소요 → GPU : Epoch 1회당 평균 57.73 sec로 단축 |

- [ ]  탐지시 낮은 정확도와 육지를 강으로 인식

| **원인** | 데이터 개수의 부족으로 인한 불필요한 인식 |
| --- | --- |
| **해결** | - 데이터 300 → 927개로 증가시켜 학습 시킨 후 불필요한 인식 확률 감소
- 강 영역 정확도 평균 78% 에서 평균 95%로 향상 |

- [ ]  위험 및 경고 상황 판별의 어려움

| **원인** | 사고 영상의 위험 상황 부재시 Object Detection만으로는 상황 판별 불가 |
| --- | --- |
| **해결** | - Sementic Segmantion을 통한 Roi 좌표를 따로 저장
- Object Detection을 통한 Bbox의 각 꼭지점 좌표 저장
- Bbox의 좌표가 Roi 좌표 안에 포함시 위험 상황으로 판별
- Bbox의 좌표가 Roi 좌표 안에 일정 영역만 포함시 경고 상황으로 판별
- Bbox의 좌표가 Roi 좌표 밖에 존재 시 안전 상황으로 판별 |

- [ ]  위험 상황을 서버에 전달은 성공 하지만 결과 영상 전달은 실패

| **원인** | 코덱에 대한 문제 였으며,  MP4V 코덱은 크롬에서 지원하지 않는 문제 |
| --- | --- |
| **해결** | - 크롬에서 지원하는 H264 코덱으로 영상 검출을 진행 하여 해결 |

- [ ]  그 외 어려웠던 점
- 라이브러리들의 버전 통일 고려하지 않고 프로젝트를 설계 했으며, 구현 시작 시 오류에 대한 원인을 찾는 것에 시간이 많이 걸렸음.
    
    이를 통해 설계 시 버전 통일의 중요성에 대해 깨닫게 되었음
    

- 아쉬웠던 점
- 데이터가 존재하지 않아서 수집과 라벨링을 수작업으로 수행해 시간이 오래걸렸다. 
데이터 라벨링을 자동으로 수행하는 모듈을 만들어 봐야겠음을 느꼈다.
