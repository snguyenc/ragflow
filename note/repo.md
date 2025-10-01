Khi fork một repo:

  - main branch của fork = branch chính trong repo fork của bạn (trên GitHub account của bạn)
  - origin/main = main branch ở remote fork của bạn
  - upstream/main = main branch ở repo gốc (original repo)

  Quy tắc làm việc:

  1. KHÔNG commit trực tiếp vào main của fork - giữ nó clean để sync với upstream
  2. Tạo branch riêng từ main để làm việc:
  git checkout main
  git checkout -b feature/my-feature
  3. Workflow chuẩn:
  # Sync main của fork với upstream
  git checkout main
  git fetch upstream
  git merge upstream/main
  git push origin main

  # Tạo branch mới để làm việc
  git checkout -b fix/my-fix

  # Làm việc, commit
  git add .
  git commit -m "Fix something"

  # Push lên fork
  git push origin fix/my-fix

  # Tạo PR từ fix/my-fix → upstream/main

  Vậy main của fork nên giữ sạch để luôn sync với repo gốc, còn làm việc thì tạo branch riêng.


    # Xóa branch fix nếu không cần nữa
  git branch -d fix/my-fix
  git push origin --delete fix/my-fix