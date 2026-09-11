# ---- powerlevel10k instant prompt (must stay at the top) ----
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="powerlevel10k/powerlevel10k"
plugins=(
  git sudo docker docker-compose python pip tmux extract colored-man-pages command-not-found
  zsh-completions zsh-autosuggestions zsh-history-substring-search
  zsh-syntax-highlighting   # must be last
)
fpath+=("$ZSH/custom/plugins/zsh-completions/src")
source "$ZSH/oh-my-zsh.sh"

# ---- history ----
HISTSIZE=100000; SAVEHIST=100000
setopt HIST_IGNORE_ALL_DUPS HIST_REDUCE_BLANKS SHARE_HISTORY INC_APPEND_HISTORY

# ---- path ----
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"

# ---- fzf (Debian package) ----
[[ -f /usr/share/doc/fzf/examples/key-bindings.zsh ]] && source /usr/share/doc/fzf/examples/key-bindings.zsh
[[ -f /usr/share/doc/fzf/examples/completion.zsh ]]   && source /usr/share/doc/fzf/examples/completion.zsh
command -v fd >/dev/null && export FZF_DEFAULT_COMMAND='fd --type f --hidden --exclude .git'

# ---- keybindings ----
bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down

# ---- aliases ----
alias ll='ls -alFh --color=auto'
alias la='ls -A'
alias grep='grep --color=auto'
alias cat='bat --paging=never --style=plain'
alias ..='cd ..'
alias hl='hl-smi'                         # Gaudi status, like nvidia-smi
alias hlw='watch -n1 hl-smi'
alias dps='docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'

# ---- Gaudi env ----
export HABANA_LOGS="$HOME/.habana_logs"

[[ -f ~/.p10k.zsh ]] && source ~/.p10k.zsh
