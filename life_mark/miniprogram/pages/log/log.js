// pages/log/log.js
Page({
  data: {
    levelOptions: [
      { label: '无', value: 0 },
      { label: '轻微', value: 1 },
      { label: '中等', value: 2 },
      { label: '超级', value: 3 }
    ],
    categories: [],
    today: ''
  },

  onLoad() {
    const date = new Date();
    const todayStr = date.getFullYear() + '-' + (date.getMonth() + 1).toString().padStart(2, '0') + '-' + date.getDate().toString().padStart(2, '0');
    const cats = getApp().getCategories();
    const categories = cats.map(c => ({ name: c.key, emoji: c.emoji, title: c.name, value: 0 }));
    this.setData({ today: todayStr, categories }, () => this.loadTodayRecord());
  },

  onShow() {
    const cats = getApp().getCategories();
    const categories = cats.map(c => ({ name: c.key, emoji: c.emoji, title: c.name, value: 0 }));
    this.setData({ categories }, () => this.loadTodayRecord());
  },

  // 兼容旧数据：布尔当作 0（无），数字 0-3 原样
  normalizeLevel(v) {
    if (v === true || v === false) return 0; // 旧「是/否」统一视为「无」
    const n = parseInt(v, 10);
    if (isNaN(n) || n < 0) return 0;
    if (n > 3) return 3;
    return n;
  },

  loadTodayRecord() {
    try {
      const allRecords = wx.getStorageSync('habit_records') || [];
      const todayRecord = allRecords.find(record => record.date === this.data.today);
      if (todayRecord) {
        const updatedCategories = this.data.categories.map(cat => ({
          ...cat,
          value: this.normalizeLevel(todayRecord[cat.name])
        }));
        this.setData({ categories: updatedCategories });
        wx.showToast({ title: '已加载今日记录', icon: 'success', duration: 1500 });
      } else {
        // 无今日记录时，显式设为默认「无」(0)
        const defaultCategories = this.data.categories.map(cat => ({ ...cat, value: 0 }));
        this.setData({ categories: defaultCategories });
      }
    } catch (e) {
      console.error('加载记录失败:', e);
    }
  },

  handleSelect(e) {
    const { category, value } = e.currentTarget.dataset;
    const level = parseInt(value, 10);
    const updatedCategories = this.data.categories.map(cat => {
      if (cat.name === category) return { ...cat, value: level };
      return cat;
    });
    this.setData({ categories: updatedCategories });
  },

  submitRecord() {
    const hasUnselected = this.data.categories.some(cat => cat.value === null || cat.value === undefined);
    if (hasUnselected) {
      wx.showToast({ title: '请完成所有记录', icon: 'none' });
      return;
    }
    const newRecord = { date: this.data.today };
    this.data.categories.forEach(cat => {
      newRecord[cat.name] = cat.value;
    });
    try {
      const allRecords = wx.getStorageSync('habit_records') || [];
      const recordIndex = allRecords.findIndex(record => record.date === this.data.today);
      if (recordIndex >= 0) {
        allRecords[recordIndex] = newRecord;
      } else {
        allRecords.push(newRecord);
      }
      wx.setStorageSync('habit_records', allRecords);
      wx.showToast({
        title: recordIndex >= 0 ? '记录更新成功' : '记录保存成功',
        icon: 'success'
      });
    } catch (e) {
      console.error('保存记录失败:', e);
      wx.showToast({ title: '保存失败', icon: 'none' });
    }
  },

  clearToday() {
    try {
      const allRecords = wx.getStorageSync('habit_records') || [];
      const newRecords = allRecords.filter(record => record.date !== this.data.today);
      wx.setStorageSync('habit_records', newRecords);
      const resetCategories = this.data.categories.map(cat => ({ ...cat, value: 0 }));
      this.setData({ categories: resetCategories });
      wx.showToast({ title: '今日记录已清空', icon: 'success' });
    } catch (e) {
      console.error('清空记录失败:', e);
    }
  }
});
