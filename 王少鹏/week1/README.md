# GitHub 连不上解决方案

在使用 Git 或访问 GitHub 时，若遇到连接失败的问题，可以按以下步骤逐一排查与解决：

---

## 1. 检查网络连接

* 首先检查是否能访问 GitHub 官网：

```bash
ping github.com
```

* 如果能 ping 通，但 git clone 失败，可能是 Git 的协议或 DNS 问题；
* 如果 ping 不通，则说明你与 GitHub 的连接被阻断，可能需要使用代理或设置 hosts。

---

## 2. 使用 SSH 连接 GitHub（推荐）

当你可以正常访问 GitHub，但使用 HTTPS 出现问题时，可以切换为 SSH：

### 步骤如下：

1. **生成 SSH 密钥（如果尚未创建）**：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. **添加 SSH 密钥到 SSH agent**：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

3. **复制公钥并添加到 GitHub**：

```bash
cat ~/.ssh/id_ed25519.pub
```

将输出的内容添加到 GitHub 账户的：

> Settings → SSH and GPG keys → New SSH key

4. **测试 SSH 是否配置成功**：

```bash
ssh -T git@github.com
```

若显示如下内容说明成功：

```
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

---

## 3. 修改 hosts（解决 ping 不通或 DNS 问题）

如果你 ping 不通 `github.com` 或 `github.global.ssl.fastly.net`，可以尝试添加或更新系统 hosts 文件。

### 操作步骤：

1. **获取 GitHub 当前 IP（国内用户可借助第三方工具或使用如下命令）**：

```bash
ping github.com
```

或者使用：

```bash
nslookup github.com
nslookup assets-cdn.github.com
```

2. **编辑 hosts 文件**：

* Windows: `C:\Windows\System32\drivers\etc\hosts`
* macOS/Linux: `/etc/hosts`

以管理员权限打开并添加如下内容（示例 IP，请使用你实际查询到的）：

```
140.82.113.4    github.com
185.199.108.154  github.global.ssl.fastly.net
```

3. **保存并刷新 DNS 缓存**：

* Windows:

```cmd
ipconfig /flushdns
```

* macOS:

```bash
sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder
```

* Linux:

```bash
sudo systemctl restart nscd
```

---

## 4. 使用代理或镜像（可选）

如果以上仍无法解决，考虑使用代理或镜像来访问 GitHub：

* 使用 VPN / 全局代理；
* 设置 Git 代理：

```bash
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

* 使用镜像网站（仅用于浏览、下载）：

| 功能          | 镜像地址                                                     |
| ----------- | -------------------------------------------------------- |
| GitHub 镜像首页 | [https://hub.fastgit.xyz](https://hub.fastgit.xyz)       |
| 加速 raw 内容   | [https://raw.gitmirror.com/](https://raw.gitmirror.com/) |

---

