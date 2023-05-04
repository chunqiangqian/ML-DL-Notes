## 强制关闭当前tensorboard端口进程

```python
lsof -i:6006

# 输出
tensorboa 3094950 ubuntu xx xxxxx xxxxxxxxxx xxxx xxxxxxxxxxxxx (LISTEN)
```

```python
kill -9 3094950
```
