from ultralytics import YOLO

model = YOLO('/Users/arshiasingh/Downloads/dataset/runs/detect/runs/microplastic/baseline_640/weights/best.pt')
metrics = model.val(split='val', plots=False)

print("=" * 45)
print("       MODEL PERFORMANCE METRICS")
print("=" * 45)

print("\n── Per-class mAP@50 ──")
class_names = ['fiber', 'film', 'fragment', 'pallet']
for name, ap50 in zip(class_names, metrics.box.ap50):
    bar = "█" * int(ap50 * 20)
    print(f"  {name:<12} {ap50:.3f}  {bar}")

print("\n── Per-class mAP@50-95 ──")
for name, ap in zip(class_names, metrics.box.maps):
    bar = "█" * int(ap * 20)
    print(f"  {name:<12} {ap:.3f}  {bar}")

print("\n── Overall ──")
print(f"  mAP@50:        {metrics.box.map50:.3f}")
print(f"  mAP@50-95:     {metrics.box.map:.3f}")
print(f"  Precision:     {metrics.box.mp:.3f}")
print(f"  Recall:        {metrics.box.mr:.3f}")
print(f"  F1 Score:      {2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr):.3f}")

print("\n── Speed (per image) ──")
print(f"  Preprocess:    {metrics.speed['preprocess']:.1f}ms")
print(f"  Inference:     {metrics.speed['inference']:.1f}ms")
print(f"  Postprocess:   {metrics.speed['postprocess']:.1f}ms")

print("\n── Model Info ──")
print(f"  Classes:       {metrics.box.nc}")
print(f"  Val images:    60")
print(f"  Val instances: {int(sum(metrics.box.ap50) / len(metrics.box.ap50) * 675):.0f} (approx)")
print("=" * 45)
