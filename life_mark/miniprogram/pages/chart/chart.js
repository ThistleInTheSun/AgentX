// pages/chart/chart.js
// 程度：无(0)、轻微(1)、中等(2)、超级(3) → 白、浅、中、浓
const LEVEL_LABELS = ['无', '轻微', '中等', '超级'];

// 按索引分配颜色（支持 5、6、7… 项）
const COLOR_PALETTES = [
  ['rgba(235,237,240,0.72)', 'rgba(255,205,210,0.72)', 'rgba(244,67,54,0.72)', 'rgba(183,28,28,0.72)'],
  ['rgba(235,237,240,0.72)', 'rgba(255,249,196,0.72)', 'rgba(255,235,59,0.72)', 'rgba(249,168,37,0.72)'],
  ['rgba(235,237,240,0.72)', 'rgba(255,224,178,0.72)', 'rgba(255,152,0,0.72)', 'rgba(230,81,0,0.72)'],
  ['rgba(235,237,240,0.72)', 'rgba(200,230,201,0.72)', 'rgba(76,175,80,0.72)', 'rgba(27,94,32,0.72)'],
  ['rgba(235,237,240,0.72)', 'rgba(187,222,251,0.72)', 'rgba(33,150,243,0.72)', 'rgba(13,71,161,0.72)'],
  ['rgba(235,237,240,0.72)', 'rgba(225,190,231,0.72)', 'rgba(156,39,176,0.72)', 'rgba(74,20,140,0.72)']
];

function categoriesWithColorScale(cats) {
  return (cats || []).map((c, i) => ({
    ...c,
    colorScale: COLOR_PALETTES[i % COLOR_PALETTES.length]
  }));
}

Page({
  data: {
    levelLabels: LEVEL_LABELS,
    chartData: {},
    categories: [],
    // 显示的天数
    days: 90,
    // 当前选中的日期、索引、类别和详情（用于高亮与详情卡）
    selectedDate: null,
    selectedIndex: null,
    selectedCategory: null,
    selectedDetails: null,
    // 每个格子的日期标签（供模板使用，避免在 WXML 中调用方法）
    cellDateLabels: [],
    // 按列组织：每列 7 格（周一到周日），从左到右为时间顺序
    chartDataColumns: {},
    monthLabels: [],
    columnIndices: [],
    stats: { totalDays: 0 },
    showSettings: false,
    showEditHistory: false,
    showAddCategory: false,
    newCategoryName: '',
    newCategoryEmoji: '📌',
    editDate: '',
    levelOptions: [
      { label: '无', value: 0 },
      { label: '轻微', value: 1 },
      { label: '中等', value: 2 },
      { label: '超级', value: 3 }
    ],
    editCategories: []
  },

  onLoad() {
    const cats = getApp().getCategories();
    this.setData({
      categories: categoriesWithColorScale(cats),
      editCategories: cats.map(c => ({ name: c.key, emoji: c.emoji, title: c.name, value: 0 }))
    }, () => this.loadAndProcessData());
  },

  onShow() {
    // 每次页面显示时重新加载数据，确保显示最新记录
    this.loadAndProcessData();
  },

  // 加载并处理数据
  loadAndProcessData() {
    try {
      // 1. 从本地存储读取所有记录
      const allRecords = wx.getStorageSync('habit_records') || [];
      
      // 2. 计算统计信息
      this.calculateStats(allRecords);
      
      // 3. 生成热力图数据
      this.generateHeatmapData(allRecords);
      
    } catch (e) {
      console.error('加载图表数据失败:', e);
      wx.showToast({ title: '加载数据失败', icon: 'none' });
    }
  },

  // 兼容旧数据：布尔转 0-3
  normalizeLevel(v) {
    if (v === true) return 2;
    if (v === false) return 0;
    const n = parseInt(v, 10);
    if (isNaN(n) || n < 0) return 0;
    if (n > 3) return 3;
    return n;
  },

  calculateStats(allRecords) {
    const categories = this.data.categories || [];
    const stats = { totalDays: allRecords.length };
    categories.forEach(cat => { stats[cat.key] = 0; });
    allRecords.forEach(record => {
      categories.forEach(cat => {
        if (this.normalizeLevel(record[cat.key]) >= 1) stats[cat.key]++;
      });
    });
    this.setData({ stats });
  },

  // 本地日期格式 YYYY-MM-DD（与 log 页一致）
  formatDateLocal(d) {
    const y = d.getFullYear();
    const m = (d.getMonth() + 1).toString().padStart(2, '0');
    const day = d.getDate().toString().padStart(2, '0');
    return `${y}-${m}-${day}`;
  },

  // 生成热力图数据
  generateHeatmapData(allRecords) {
    const { days, categories } = this.data;
    if (!categories || categories.length === 0) {
      this.setData({
        chartData: {},
        cellDateLabels: [],
        chartDataColumns: {},
        monthLabels: [],
        columnIndices: []
      });
      return;
    }

    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(endDate.getDate() - days + 1);

    const emptyArray = new Array(days).fill(0);
    const chartData = {};
    categories.forEach(cat => { chartData[cat.key] = [...emptyArray]; });

    const dateToIndex = {};
    const cellDateLabels = [];
    for (let i = 0; i < days; i++) {
      const currentDate = new Date(startDate);
      currentDate.setDate(startDate.getDate() + i);
      const dateStr = this.formatDateLocal(currentDate);
      dateToIndex[dateStr] = i;
      cellDateLabels.push((currentDate.getMonth() + 1) + '/' + currentDate.getDate());
    }

    allRecords.forEach(record => {
      const index = dateToIndex[record.date];
      if (index !== undefined) {
        categories.forEach(cat => {
          chartData[cat.key][index] = this.normalizeLevel(record[cat.key]);
        });
      }
    });

    const numColumns = Math.ceil(days / 7);
    const columnIndices = Array.from({ length: numColumns }, (_, i) => i);
    const monthLabels = [];
    const chartDataColumns = {};
    categories.forEach(cat => { chartDataColumns[cat.key] = []; });

    let lastMonth = -1;
    for (let col = 0; col < numColumns; col++) {
      const firstDayIndex = col * 7;
      const firstDate = new Date(startDate);
      firstDate.setDate(startDate.getDate() + firstDayIndex);
      const curMonth = firstDate.getMonth();
      monthLabels.push(col === 0 || curMonth !== lastMonth ? (curMonth + 1) + '月' : '');
      lastMonth = curMonth;

      categories.forEach(cat => {
        const columnCells = [];
        for (let row = 0; row < 7; row++) {
          const index = col * 7 + row;
          if (index >= days) {
            columnCells.push({ value: null, dateLabel: '', index: -1 });
          } else {
            columnCells.push({
              value: chartData[cat.key][index],
              dateLabel: cellDateLabels[index],
              index: index
            });
          }
        }
        chartDataColumns[cat.key].push(columnCells);
      });
    }

    this.setData({ chartData, cellDateLabels, chartDataColumns, monthLabels, columnIndices });
  },

  // 点击热力图单元格
  onCellTap(e) {
    const { category, index } = e.currentTarget.dataset;
    const { days } = this.data;
    const indexNum = parseInt(index, 10);

    const endDate = new Date();
    const targetDate = new Date();
    targetDate.setDate(endDate.getDate() - (days - 1 - indexNum));
    const dateStr = this.formatDateLocal(targetDate);

    let details = null;
    try {
      const allRecords = wx.getStorageSync('habit_records') || [];
      const record = allRecords.find(r => r.date === dateStr);
      const categories = this.data.categories || [];
      if (record && categories.length > 0) {
        details = { date: dateStr };
        categories.forEach(cat => {
          const level = this.normalizeLevel(record[cat.key]);
          details[cat.key] = LEVEL_LABELS[level];
          details[cat.key + 'Level'] = level;
        });
      }
    } catch (e) {
      console.error('获取详情失败:', e);
    }

    this.setData({
      selectedDate: dateStr,
      selectedIndex: indexNum,
      selectedCategory: category,
      selectedDetails: details
    });
  },

  // 切换显示天数
  switchDays(e) {
    const days = parseInt(e.currentTarget.dataset.days, 10);
    this.setData({
      days,
      selectedDate: null,
      selectedIndex: null,
      selectedCategory: null,
      selectedDetails: null
    }, () => {
      this.loadAndProcessData();
    });
  },

  openSettings() {
    this.setData({ showSettings: true });
  },

  preventClose() {},

  closeSettings() {
    this.setData({ showSettings: false });
  },

  onExportData() {
    this.exportData();
    this.setData({ showSettings: false });
  },

  exportData() {
    try {
      const allRecords = wx.getStorageSync('habit_records') || [];
      const dataStr = JSON.stringify(allRecords, null, 2);
      wx.setClipboardData({
        data: dataStr,
        success: () => {
          this.setData({ showSettings: false });
          wx.showModal({
            title: '数据已复制',
            content: `所有记录已复制到剪贴板，共${allRecords.length}条记录。\n\n可粘贴到备忘录或文件中备份。`,
            showCancel: false
          });
        }
      });
    } catch (e) {
      console.error('导出数据失败:', e);
      wx.showToast({ title: '导出失败', icon: 'none' });
    }
  },

  openEditHistory() {
    const d = new Date();
    const editDate = d.getFullYear() + '-' + (d.getMonth() + 1).toString().padStart(2, '0') + '-' + d.getDate().toString().padStart(2, '0');
    const categories = this.data.categories || [];
    const editCategories = categories.map(c => ({ name: c.key, emoji: c.emoji, title: c.name, value: 0 }));
    this.setData({ showSettings: false, showEditHistory: true, editDate, editCategories }, () => {
      this.loadEditRecordForDate(editDate);
    });
  },

  closeEditHistory() {
    this.setData({ showEditHistory: false });
  },

  loadEditRecordForDate(dateStr) {
    const allRecords = wx.getStorageSync('habit_records') || [];
    const record = allRecords.find(r => r.date === dateStr);
    const editCategories = (this.data.editCategories || []).map(cat => ({
      ...cat,
      value: record ? this.normalizeLevel(record[cat.name]) : 0
    }));
    this.setData({ editCategories });
  },

  onEditDateChange(e) {
    const editDate = e.detail.value;
    this.setData({ editDate }, () => {
      this.loadEditRecordForDate(editDate);
    });
  },

  onEditSelect(e) {
    const { category, value } = e.currentTarget.dataset;
    const level = parseInt(value, 10);
    const editCategories = this.data.editCategories.map(cat => {
      if (cat.name === category) return { ...cat, value: level };
      return cat;
    });
    this.setData({ editCategories });
  },

  saveEditRecord() {
    const { editDate, editCategories } = this.data;
    if (!editDate) {
      wx.showToast({ title: '请选择日期', icon: 'none' });
      return;
    }
    const newRecord = { date: editDate };
    editCategories.forEach(cat => { newRecord[cat.name] = cat.value; });
    try {
      const allRecords = wx.getStorageSync('habit_records') || [];
      const idx = allRecords.findIndex(r => r.date === editDate);
      if (idx >= 0) allRecords[idx] = newRecord;
      else allRecords.push(newRecord);
      wx.setStorageSync('habit_records', allRecords);
      this.setData({ showEditHistory: false });
      this.loadAndProcessData();
      wx.showToast({ title: '已保存', icon: 'success' });
    } catch (e) {
      console.error('保存失败:', e);
      wx.showToast({ title: '保存失败', icon: 'none' });
    }
  },

  // 清空所有数据（谨慎操作）
  clearAllData() {
    wx.showModal({
      title: '警告',
      content: '这将永久删除所有记录，且无法恢复。确定要继续吗？',
      confirmColor: '#e74c3c',
      success: (res) => {
        if (res.confirm) {
          try {
            wx.removeStorageSync('habit_records');
            const categories = this.data.categories || [];
            const chartData = {};
            const chartDataColumns = {};
            const stats = { totalDays: 0 };
            categories.forEach(cat => {
              chartData[cat.key] = [];
              chartDataColumns[cat.key] = [];
              stats[cat.key] = 0;
            });
            this.setData({
              chartData,
              chartDataColumns,
              monthLabels: [],
              columnIndices: [],
              stats,
              selectedDate: null,
              selectedIndex: null,
              selectedCategory: null,
              selectedDetails: null,
              cellDateLabels: []
            });
            wx.showToast({ title: '数据已清空', icon: 'success' });
          } catch (e) {
            console.error('清空数据失败:', e);
          }
        }
      }
    });
  },

  // 打卡项目：显示添加表单
  showAddCategoryForm() {
    this.setData({ showAddCategory: true, newCategoryName: '', newCategoryEmoji: '📌' });
  },

  cancelAddCategory() {
    this.setData({ showAddCategory: false, newCategoryName: '', newCategoryEmoji: '📌' });
  },

  onNewCategoryNameInput(e) {
    this.setData({ newCategoryName: (e.detail && e.detail.value) || '' });
  },

  onNewCategoryEmojiInput(e) {
    this.setData({ newCategoryEmoji: (e.detail && e.detail.value) || '📌' });
  },

  confirmAddCategory() {
    const name = (this.data.newCategoryName || '').trim();
    const emoji = (this.data.newCategoryEmoji || '📌').trim() || '📌';
    if (!name) {
      wx.showToast({ title: '请输入项目名称', icon: 'none' });
      return;
    }
    const key = 'cat_' + Date.now();
    const raw = getApp().getCategories();
    const next = raw.concat([{ key, name, emoji }]);
    getApp().saveCategories(next);
    const categories = categoriesWithColorScale(next);
    this.setData({
      categories,
      editCategories: next.map(c => ({ name: c.key, emoji: c.emoji, title: c.name, value: 0 })),
      showAddCategory: false,
      newCategoryName: '',
      newCategoryEmoji: '📌'
    }, () => {
      this.loadAndProcessData();
      wx.showToast({ title: '已添加', icon: 'success' });
    });
  },

  removeCategory(e) {
    const key = e.currentTarget.dataset.key;
    const categories = this.data.categories || [];
    if (categories.length <= 1) {
      wx.showModal({
        title: '删除',
        content: '删除后将没有打卡项目，记录页将无法记录。确定删除？',
        success: (res) => res.confirm && this.doRemoveCategory(key)
      });
      return;
    }
    wx.showModal({
      title: '删除打卡项目',
      content: '将同时删除该项目在所有日期下的记录，且无法恢复。确定删除？',
      confirmColor: '#e74c3c',
      success: (res) => res.confirm && this.doRemoveCategory(key)
    });
  },

  doRemoveCategory(key) {
    const categories = this.data.categories || [];
    const next = categories.filter(c => c.key !== key).map(({ key: k, name, emoji }) => ({ key: k, name, emoji }));
    getApp().saveCategories(next);
    try {
      const allRecords = wx.getStorageSync('habit_records') || [];
      allRecords.forEach(record => {
        if (record[key] !== undefined) delete record[key];
      });
      wx.setStorageSync('habit_records', allRecords);
    } catch (e) {}
    const newCategories = categoriesWithColorScale(next);
    this.setData({
      categories: newCategories,
      editCategories: next.map(c => ({ name: c.key, emoji: c.emoji, title: c.name, value: 0 }))
    }, () => {
      this.loadAndProcessData();
      wx.showToast({ title: '已删除', icon: 'success' });
    });
  }
});