---
title: "Xây dựng hệ thống AI thực dụng: Từ ý tưởng model đến workflow ngoài đời thực"
description: "Một hướng dẫn kỹ thuật về cách biến model thành workflow AI đáng tin cậy bằng routing, validation, tools, human review và vận hành."
pubDatetime: "2026-09-01T08:00:00.000Z"
locale: vi
author: Michael
featured: false
tags:
  - AI Systems
  - Workflow Automation
  - Computer Vision
  - Engineering
categories:
  - AI
  - Technical
  - Product
---

Một model không phải là một sản phẩm. Nó chỉ là một thành phần trong hệ thống phải tiếp nhận input không hoàn hảo, ra quyết định dưới sự không chắc chắn và để lại bằng chứng có thể kiểm tra.

Qua các case công khai về document intelligence, computer vision, edge translation và AI workflow automation, một bài học kỹ thuật lặp lại rất rõ: workflow quan trọng không kém model.

## Bắt đầu từ workflow

Trước khi chọn model, hãy định nghĩa quy trình:

```txt
input -> phân loại -> định tuyến -> suy luận -> kiểm tra -> review -> sửa -> xuất kết quả
```

Cách này khiến failure trở nên rõ ràng. Kết quả OCR có confidence thấp có thể được chuyển sang provider thứ hai. Dự đoán computer vision có thể được kiểm tra bằng hình học hoặc business rule. Agent chỉ được gọi tool khi state machine cho phép.

## Routing là quyết định sản phẩm

Các input khác nhau cần các con đường khác nhau. Một hóa đơn rõ ràng, một biểu mẫu chụp bằng điện thoại và một bản scan bị hỏng không nhất thiết nên dùng cùng cấu hình OCR. Routing có thể dựa trên loại tài liệu, ngôn ngữ, confidence, ngân sách latency và chi phí.

Với AI agent cũng vậy. Model nhỏ có thể làm phân loại, còn model reasoning mạnh hơn xử lý ca mơ hồ. Cách này thường rẻ và dễ vận hành hơn việc gửi mọi request tới model lớn nhất.

## Validation quan trọng hơn confidence mù quáng

Điểm số của model không phải bằng chứng. Hệ thống thực tế cần required fields, range, schema, tính nhất quán giữa các field, phát hiện trùng lặp và domain rules. Output nên mang provenance: model nào chạy, version nào, tool nào được gọi và field nào đã được review.

## Human review không phải thất bại

Human review là một control surface. Nó cần có chọn lọc, dễ hiểu và dễ hoàn thành. Reviewer cần thấy dữ liệu nguồn, giá trị được extract, tín hiệu không chắc chắn và cách sửa. Correction chỉ nên trở thành dữ liệu training hoặc evaluation sau khi được gắn nhãn và quản trị.

## Đo cả hệ thống

Accuracy một mình che giấu nhiều trade-off. Hãy đo task success, chất lượng abstain, latency, cost, retry rate, review burden và thời gian recovery. Một hệ thống kém chính xác hơn một chút nhưng predictable và repairable có thể hữu ích hơn model đứng đầu benchmark nhưng failure khó hiểu.

Quy tắc thực dụng là: chọn hệ thống nhỏ nhất có thể hoàn thành workflow an toàn, rồi cải thiện model ở đúng bottleneck mà bằng chứng chỉ ra. AI engineering không chỉ là chọn model; đó là thiết kế một vòng lặp có thể làm việc, thất bại, giải thích và phục hồi.
