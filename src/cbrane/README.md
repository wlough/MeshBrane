# **CppBrane**

# Packages

-armadillo
-

## Update system/mirrors/etc...
Update mirror lists:
```bash
reflector --fastest 30 --latest 30 --number 10 --download-timeout 30 \
--country us,ca,mx --sort age --verbose --save /etc/pacman.d/mirrorlist
```

Download fresh package databases from the server, upgrade installed packages from arch and aur repos
```bash
yay
```
