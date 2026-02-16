---
name: doc-auto-push
description: This skill should be used when markdown documentation files (*.md) in the theory/ directory are created or modified. It automatically commits and pushes changes to the remote repository with meaningful commit messages.
disable: false
---

# Documentation Auto-Push Skill

This skill automates the process of committing and pushing documentation changes to the remote Git repository.

## When to Use

This skill is triggered automatically when:
- Markdown files (*.md) in `theory/` directory are created or modified
- Documentation updates are complete
- User explicitly requests to push documentation changes

## Workflow

### Step 1: Detect Changed Documentation Files

Check Git status for modified markdown files:
```bash
git status theory/
```

### Step 2: Stage Documentation Changes

Add all modified documentation files:
```bash
git add theory/**/*.md
```

### Step 3: Generate Meaningful Commit Message

Create a descriptive commit message based on:
- Which files were modified
- The type of changes (creation, update, refactor)
- Brief description of the changes

**Message format**:
```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `docs`: Documentation creation or major update
- `refactor`: Documentation restructuring or rewriting
- `fix`: Fix typos or errors in documentation
- `style`: Formatting changes only

**Example**:
```
docs: 重构第4章为教学资料风格

- 增加问题驱动的讲解方式
- 添加循序渐进的算法演进历史
- 补充丰富的代码示例和对比表格
- 新增实践指南和学习路径建议
- 保持与前三章一致的教学风格
```

### Step 4: Commit Changes

```bash
git commit -m "<commit_message>"
```

### Step 5: Push to Remote

```bash
git push origin main
```

### Step 6: Verify Push Success

Check that the push was successful and report the commit hash.

## Best Practices

### 1. Meaningful Commit Messages

**Good**:
```
docs: 完善贝尔曼方程推导过程

- 添加详细的数学推导步骤
- 补充几何直观解释
- 增加推荐系统应用实例
```

**Bad**:
```
update docs
```

### 2. Atomic Commits

Group related changes together:
- If updating multiple chapters on the same topic → one commit
- If fixing typos across files → separate commit
- If major refactoring → separate commit

### 3. Review Before Push

Always review changes before pushing:
```bash
git diff theory/
```

### 4. Handle Conflicts

If push fails due to conflicts:
1. Pull latest changes: `git pull origin main`
2. Resolve conflicts
3. Commit merge
4. Push again

## Common Scenarios

### Scenario 1: New Chapter Created

**Trigger**: User creates a new markdown file like `05_new_topic.md`

**Action**:
```bash
git add theory/foundations/05_new_topic.md
git commit -m "docs: 新增第5章 - <主题>

- 章节结构
- 核心内容
- 学习目标"
git push origin main
```

### Scenario 2: Chapter Content Updated

**Trigger**: User modifies existing chapter content

**Action**:
```bash
git add theory/foundations/04_rl_evolution_to_onerec.md
git commit -m "refactor: 重构第4章为教学资料风格

- 问题驱动讲解
- 算法演进详解
- 实践指南"
git push origin main
```

### Scenario 3: Multiple Files Modified

**Trigger**: User updates several related files

**Action**:
```bash
git add theory/foundations/*.md
git commit -m "docs: 统一文档风格

- 所有章节采用一致的结构
- 标准化代码示例格式
- 优化图表展示"
git push origin main
```

### Scenario 4: Fix Typos

**Trigger**: User fixes typos or formatting issues

**Action**:
```bash
git add theory/foundations/03_bellman_equations_detailed.md
git commit -m "fix: 修正第3章公式错误和排版问题"
git push origin main
```

## Error Handling

### Push Rejected (Non-Fast-Forward)

**Problem**: Remote has changes not in local

**Solution**:
```bash
git pull --rebase origin main
# Resolve conflicts if any
git push origin main
```

### Merge Conflicts

**Problem**: Conflicting changes in same file

**Solution**:
1. Identify conflicting files
2. Manually resolve conflicts
3. Stage resolved files: `git add <file>`
4. Continue: `git rebase --continue` or `git commit`
5. Push: `git push origin main`

### Authentication Failed

**Problem**: Git credentials expired or incorrect

**Solution**:
1. Check SSH keys: `ssh -T git@github.com`
2. Or use HTTPS with token
3. Update credentials as needed

## Integration with Other Skills

This skill works well with:
- **doc-validator**: Validate markdown syntax before push
- **link-checker**: Check all internal links work
- **spell-checker**: Fix typos before committing

## Configuration

### Custom Commit Message Template

Create `.gitmessage` template:
```
<type>: <subject>

## 变更内容
- 

## 影响范围
- 

## 相关文档
- 
```

Use with:
```bash
git config commit.template .gitmessage
```

### Auto-Push on Save (Optional)

**Not recommended** for documentation (prefer manual review), but possible:
```bash
# Git hook: .git/hooks/post-commit
#!/bin/bash
if [[ $(git log -1 --pretty=%B) == docs:* ]]; then
    git push origin main
fi
```

## Safety Checks

Before pushing, verify:
- [ ] All code examples are syntactically correct
- [ ] All internal links are valid
- [ ] No sensitive information (API keys, passwords) included
- [ ] Markdown renders correctly
- [ ] Images/diagrams load properly

## Examples

### Example 1: Major Chapter Update

```bash
# User heavily modifies chapter 4
AI detects changes to: theory/foundations/04_rl_evolution_to_onerec.md

# Execute workflow
git add theory/foundations/04_rl_evolution_to_onerec.md
git commit -m "refactor: 重构第4章为教学资料风格

- 增加问题驱动的讲解方式
- 添加循序渐进的算法演进历史（REINFORCE→PPO→GRPO→ECPO）
- 补充丰富的代码示例和对比表格
- 新增实践指南和学习路径建议
- 保持与前三章一致的教学风格

主要改进：
- 第2章：推荐系统面临的三大挑战详解
- 第4章：Value-Based方法失效的根本原因
- 第5章：Policy-Based方法的完整演进
- 第7章：实践算法选择决策树
- 第8章：项目驱动的学习路径

文件大小：1758行（相比之前的389行）"

git push origin main
# Output: Successfully pushed to origin/main
```

### Example 2: Quick Typo Fix

```bash
# User fixes a formula typo
git add theory/foundations/03_bellman_equations_detailed.md
git commit -m "fix: 修正贝尔曼最优方程的LaTeX公式"
git push origin main
```

### Example 3: README Update

```bash
# User updates main README to reflect new chapter
git add README.md
git commit -m "docs: 更新README以反映第4章的改进

- 增强核心洞察部分
- 更新学习路线图
- 添加最新论文列表（GRPO, OneRec）"
git push origin main
```

## Quick Reference

**Basic Flow**:
```
Modified *.md → git add → git commit → git push → Verify
```

**Commit Message Template**:
```
<type>: <brief summary>

<detailed description>
- bullet point 1
- bullet point 2

<optional footer>
```

**Common Commands**:
```bash
# Check status
git status

# Stage specific file
git add theory/foundations/04_rl_evolution_to_onerec.md

# Stage all theory docs
git add theory/**/*.md

# Commit with message
git commit -m "docs: update chapter 4"

# Push to main
git push origin main

# Check push result
git log -1
```

---

## Notes

- This skill focuses on `theory/` directory documentation
- Always review changes before pushing
- Use meaningful commit messages for better collaboration
- Keep commits atomic and focused
- Don't push incomplete or broken documentation

**Related Skills**:
- `skill-creator`: Learn how to create new skills
- `pdf-reader`: Extract content from PDF references
