class AiCanDashboard {
    constructor() {
        this.config = this.loadConfig();
        this.initializeElements();
        this.attachEventListeners();
        this.updateStatus();
        this.log('Dashboard initialized successfully');
    }

    initializeElements() {
        // Get all DOM elements
        this.elements = {
            task: document.getElementById('task'),
            runTask: document.getElementById('runTask'),
            taskStatus: document.getElementById('taskStatus'),
            
            notionPageId: document.getElementById('notionPageId'),
            notionStatus: document.getElementById('notionStatus'),
            updateNotion: document.getElementById('updateNotion'),
            notionStatusDiv: document.getElementById('notionStatus'),
            
            githubToken: document.getElementById('githubToken'),
            githubRepo: document.getElementById('githubRepo'),
            notionToken: document.getElementById('notionToken'),
            saveConfig: document.getElementById('saveConfig'),
            
            systemInfo: document.getElementById('systemInfo'),
            githubStatus: document.getElementById('githubStatus'),
            notionApiStatus: document.getElementById('notionApiStatus'),
            lastAction: document.getElementById('lastAction'),
            checkStatus: document.getElementById('checkStatus'),
            
            logs: document.getElementById('logs'),
            clearLogs: document.getElementById('clearLogs')
        };

        // Load saved configuration into form fields
        if (this.config.githubRepo) {
            this.elements.githubRepo.value = this.config.githubRepo;
        }
    }

    attachEventListeners() {
        this.elements.runTask.addEventListener('click', () => this.runTask());
        this.elements.updateNotion.addEventListener('click', () => this.updateNotionStatus());
        this.elements.saveConfig.addEventListener('click', () => this.saveConfiguration());
        this.elements.checkStatus.addEventListener('click', () => this.updateStatus());
        this.elements.clearLogs.addEventListener('click', () => this.clearLogs());
    }

    loadConfig() {
        const saved = localStorage.getItem('aicanConfig');
        return saved ? JSON.parse(saved) : {
            githubToken: '',
            githubRepo: '',
            notionToken: ''
        };
    }

    saveConfig() {
        localStorage.setItem('aicanConfig', JSON.stringify(this.config));
    }

    saveConfiguration() {
        this.config.githubToken = this.elements.githubToken.value;
        this.config.githubRepo = this.elements.githubRepo.value;
        this.config.notionToken = this.elements.notionToken.value;
        
        this.saveConfig();
        this.log('✅ Configuration saved successfully');
        this.updateStatus();
        
        // Clear password fields for security
        this.elements.githubToken.value = '';
        this.elements.notionToken.value = '';
        
        this.showStatus(this.elements.saveConfig.parentElement, 'success', 'Configuration saved!');
    }

    async runTask() {
        const task = this.elements.task.value;
        const notionPageId = this.elements.notionPageId.value;

        if (!this.config.githubToken || !this.config.githubRepo) {
            this.showStatus(this.elements.taskStatus, 'error', 'Please configure GitHub settings first');
            return;
        }

        this.log(`🚀 Triggering task: ${task}`);
        this.setButtonLoading(this.elements.runTask, true);
        this.showStatus(this.elements.taskStatus, 'loading', `Executing ${task}...`);

        try {
            const response = await this.triggerWorkflow(task, notionPageId);
            
            if (response.ok) {
                this.log(`✅ Successfully triggered ${task}`);
                this.showStatus(this.elements.taskStatus, 'success', `${task} started successfully!`);
                this.config.lastAction = `${task} - ${new Date().toLocaleString()}`;
                this.saveConfig();
                this.updateStatus();
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            this.log(`❌ Error: ${error.message}`);
            this.showStatus(this.elements.taskStatus, 'error', `Failed to run ${task}: ${error.message}`);
        } finally {
            this.setButtonLoading(this.elements.runTask, false);
        }
    }

    async triggerWorkflow(task, notionPageId = '') {
        const [owner, repo] = this.config.githubRepo.split('/');
        const url = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/dispatch.yml/dispatches`;
        
        const payload = {
            ref: 'main',
            inputs: {
                task: task,
                ...(notionPageId && { notion_page: notionPageId })
            }
        };

        return fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `token ${this.config.githubToken}`,
                'Accept': 'application/vnd.github.v3+json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
    }

    async updateNotionStatus() {
        const pageId = this.elements.notionPageId.value;
        const status = this.elements.notionStatus.value;

        if (!pageId) {
            this.showStatus(this.elements.notionStatusDiv, 'error', 'Please enter a Notion page ID');
            return;
        }

        if (!this.config.notionToken) {
            this.showStatus(this.elements.notionStatusDiv, 'error', 'Please configure Notion token first');
            return;
        }

        this.log(`📝 Updating Notion page ${pageId} to status: ${status}`);
        this.setButtonLoading(this.elements.updateNotion, true);
        this.showStatus(this.elements.notionStatusDiv, 'loading', 'Updating Notion...');

        try {
            const response = await this.updateNotionPage(pageId, status);
            
            if (response.ok) {
                this.log(`✅ Notion page updated successfully`);
                this.showStatus(this.elements.notionStatusDiv, 'success', 'Status updated successfully!');
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            this.log(`❌ Notion error: ${error.message}`);
            this.showStatus(this.elements.notionStatusDiv, 'error', `Failed to update: ${error.message}`);
        } finally {
            this.setButtonLoading(this.elements.updateNotion, false);
        }
    }

    async updateNotionPage(pageId, status) {
        const url = `https://api.notion.com/v1/pages/${pageId}`;
        
        const payload = {
            properties: {
                Status: {
                    status: {
                        name: status
                    }
                }
            }
        };

        return fetch(url, {
            method: 'PATCH',
            headers: {
                'Authorization': `Bearer ${this.config.notionToken}`,
                'Notion-Version': '2022-06-28',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
    }

    async updateStatus() {
        let githubOk = false;
        let notionOk = false;

        // Test GitHub API
        if (this.config.githubToken && this.config.githubRepo) {
            try {
                const [owner, repo] = this.config.githubRepo.split('/');
                const response = await fetch(`https://api.github.com/repos/${owner}/${repo}`, {
                    headers: { 'Authorization': `token ${this.config.githubToken}` }
                });
                githubOk = response.ok;
            } catch (error) {
                githubOk = false;
            }
        }

        // Test Notion API (basic auth check)
        if (this.config.notionToken) {
            try {
                const response = await fetch('https://api.notion.com/v1/users/me', {
                    headers: {
                        'Authorization': `Bearer ${this.config.notionToken}`,
                        'Notion-Version': '2022-06-28'
                    }
                });
                notionOk = response.ok;
            } catch (error) {
                notionOk = false;
            }
        }

        // Update status indicators
        this.elements.githubStatus.textContent = githubOk ? 'Connected ✅' : 'Not configured ❌';
        this.elements.githubStatus.style.color = githubOk ? '#38a169' : '#e53e3e';
        
        this.elements.notionApiStatus.textContent = notionOk ? 'Connected ✅' : 'Not configured ❌';
        this.elements.notionApiStatus.style.color = notionOk ? '#38a169' : '#e53e3e';
        
        this.elements.lastAction.textContent = this.config.lastAction || 'None';
        
        this.log(`🔄 Status updated - GitHub: ${githubOk ? 'OK' : 'FAIL'}, Notion: ${notionOk ? 'OK' : 'FAIL'}`);
    }

    showStatus(element, type, message) {
        element.className = `status ${type}`;
        element.textContent = message;
        element.style.display = 'block';
        
        // Auto-hide after 5 seconds for success messages
        if (type === 'success') {
            setTimeout(() => {
                element.style.display = 'none';
            }, 5000);
        }
    }

    setButtonLoading(button, loading) {
        if (loading) {
            button.disabled = true;
            button.dataset.originalText = button.textContent;
            button.textContent = 'Loading...';
        } else {
            button.disabled = false;
            button.textContent = button.dataset.originalText || button.textContent;
        }
    }

    log(message) {
        const timestamp = new Date().toLocaleTimeString();
        const logMessage = `[${timestamp}] ${message}\n`;
        this.elements.logs.textContent += logMessage;
        this.elements.logs.scrollTop = this.elements.logs.scrollHeight;
    }

    clearLogs() {
        this.elements.logs.textContent = 'Logs cleared.\n\nReady for new actions...\n';
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new AiCanDashboard();
});

// Expose some methods for debugging
window.AiCanDashboard = AiCanDashboard;