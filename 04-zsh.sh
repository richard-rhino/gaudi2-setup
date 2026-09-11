#!/usr/bin/env bash
# zsh + oh-my-zsh + plugins + powerlevel10k for the current user. No sudo needed.
# Run as yourself:  bash ~/setup/04-zsh.sh
set -euo pipefail
command -v zsh >/dev/null || { echo "zsh not installed yet; run 01-base.sh first"; exit 1; }

# oh-my-zsh (unattended, don't switch shell or start zsh)
if [[ ! -d ~/.oh-my-zsh ]]; then
  RUNZSH=no CHSH=no KEEP_ZSHRC=yes sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
fi
ZC=~/.oh-my-zsh/custom
clone() { [[ -d "$2" ]] || git clone --depth=1 "$1" "$2"; }
clone https://github.com/zsh-users/zsh-autosuggestions        $ZC/plugins/zsh-autosuggestions
clone https://github.com/zsh-users/zsh-syntax-highlighting    $ZC/plugins/zsh-syntax-highlighting
clone https://github.com/zsh-users/zsh-completions             $ZC/plugins/zsh-completions
clone https://github.com/zsh-users/zsh-history-substring-search $ZC/plugins/zsh-history-substring-search
clone https://github.com/romkatv/powerlevel10k.git             $ZC/themes/powerlevel10k

[[ -f ~/.zshrc ]] && cp ~/.zshrc ~/.zshrc.bak.$(date +%s)
cp ~/setup/zshrc ~/.zshrc
[[ -f ~/.p10k.zsh ]] || cp ~/setup/p10k.zsh ~/.p10k.zsh
echo "=== 04-zsh done === log out and back in (login shell was set by 01-base.sh), or run: exec zsh"
