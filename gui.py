#
# A GUI for ps-auxw's CLI-P (commandline driven semantic image search using OpenAI's CLIP)
#

import sys
from io import StringIO
import contextlib

from PyQt5.QtCore import (
    pyqtSignal,
    Qt,
    QItemSelectionModel,
    QTimer,
)
from PyQt5.QtGui import (
    QStandardItemModel, QStandardItem,
    QImage, QPixmap,
)
from PyQt5.QtWidgets import (
    qApp,
    QApplication, QMainWindow, QWidget,
    QSizePolicy,
    QHBoxLayout, QVBoxLayout, QTabWidget,
    QComboBox, QLabel, QPushButton, QTextEdit,
    QTableView,
)

# Load delayed, so the GUI is already visible,
# as this may take a long time.
query_index = None

class HistoryComboBox(QComboBox):
    pageChangeRequested = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(HistoryComboBox, self).__init__(parent)
        self.isShowingPopup = False
        self._defaultButton = None

    def defaultButton(self):
        return self._defaultButton

    def setDefaultButton(self, button):
        self._defaultButton = button

    def showPopup(self):
        super(HistoryComboBox, self).showPopup()
        self.isShowingPopup = True

    def hidePopup(self):
        self.isShowingPopup = False
        super(HistoryComboBox, self).hidePopup()

    def keyPressEvent(self, ev):
        key = ev.key()
        # On Return (Here, Enter is located on the key pad, instead!),
        # activate the associated default button.
        # Necessary in non-dialogs.
        if key == Qt.Key_Return and self._defaultButton != None:
            # Propagate further, first.
            # Necessary so the user input still gets added to the list.
            super(HistoryComboBox, self).keyPressEvent(ev)
            # Then, activate the default button.
            self._defaultButton.click()
        # On up/down, ensure the popup opens.
        elif (key == Qt.Key_Up or key == Qt.Key_Down) and not self.isShowingPopup:
            self.showPopup()
            # Don't prevent default handling of the key press.
            super(HistoryComboBox, self).keyPressEvent(ev)
        # On PageUp/Down, emit a signal
        # (so this can control, e.g., scrolling of the console log).
        elif key == Qt.Key_PageUp or key == Qt.Key_PageDown:
            self.pageChangeRequested.emit(key == Qt.Key_PageUp)
            # Don't prevent default handling of the key press.
            super(HistoryComboBox, self).keyPressEvent(ev)
        # Otherwise, propagate key press further.
        else:
            super(HistoryComboBox, self).keyPressEvent(ev)

class MainWindow(QMainWindow):
    class OurTabPage(QWidget):
        resized = pyqtSignal()
        def __init__(self, parent=None):
            super(MainWindow.OurTabPage, self).__init__(parent)
        def resizeEvent(self, ev):
            super(MainWindow.OurTabPage, self).resizeEvent(ev)
            self.resized.emit()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.search = None
        self.searchResultSelected = None

        # TODO: Take from config db.
        self.resize(1600, 900)

        centralWidget = QWidget()
        centralVBox = QVBoxLayout(centralWidget)

        self.tabWidget = QTabWidget()


        # Page 1: Console
        self.consoleTabPage = QWidget()
        consoleVBox = QVBoxLayout(self.consoleTabPage)

        self.infoLabel = QLabel(
            "ps-auxw says, \"CLI-P is commandline driven semantic image search using OpenAI's CLIP\"\n"
            "canvon says, \"This is a GUI for ps-auxw's CLI-P\"")
        self.searchOutput = QTextEdit()
        self.searchOutput.setReadOnly(True)

        consoleVBox.addWidget(self.infoLabel)
        consoleVBox.addWidget(self.searchOutput)
        self.tabWidget.addTab(self.consoleTabPage, "&1 Console")


        # Page 2: Images
        self.imagesTabPage = self.OurTabPage()
        self.imagesTabPage.resized.connect(self.imagesTabPageResized)
        imagesVBox = QVBoxLayout(self.imagesTabPage)

        self.imageLabel = QLabel()
        self.imagesTableView = QTableView()
        self.imagesTableView.setEditTriggers(QTableView.NoEditTriggers)
        self.imagesTableView.activated.connect(self.searchResultsActivated)

        imagesVBox.addWidget(self.imageLabel)
        imagesVBox.addWidget(self.imagesTableView)
        self.tabWidget.addTab(self.imagesTabPage, "&2 Images")


        self.searchHint = QLabel()


        # Search input box & go button
        inputHBox = QHBoxLayout()

        self.searchInputLabel = QLabel("&Search")
        self.searchInputLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.searchInput = HistoryComboBox()
        self.searchInput.setEditable(True)
        self.searchInput.pageChangeRequested.connect(self.searchInputPageChangeRequested)
        #
        # This fired too often, but we only want to search
        # when the user finally hits return...
        #self.searchInput.activated.connect(self.handleSearchInput)
        #
        self.searchInputLabel.setBuddy(self.searchInput)

        self.searchInputButton = QPushButton()
        #
        # Doesn't work without a Dialog:
        #self.searchInputButton.setAutoDefault(True)
        #self.searchInputButton.setDefault(True)
        # ..so, do this instead:
        self.searchInput.setDefaultButton(self.searchInputButton)
        #
        self.searchInputButton.setText("&Go")
        self.searchInputButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.searchInputButton.clicked.connect(self.handleSearchInput)

        inputHBox.addWidget(self.searchInputLabel)
        inputHBox.addWidget(self.searchInput)
        inputHBox.addWidget(self.searchInputButton)


        centralVBox.addWidget(self.tabWidget)
        centralVBox.addWidget(self.searchHint)
        centralVBox.addLayout(inputHBox)

        self.setCentralWidget(centralWidget)
        self.searchInput.setFocus()

        self.createSearchResultsModel()

    def imagesTabPageResized(self):
        contents = self.imagesTabPage.contentsRect()
        self.imageLabel.setMaximumSize(contents.width(), contents.height() * 8 / 10)  # 80%
        self.imageLabel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

    def searchInputPageChangeRequested(self, pageUp):
        w = self.tabWidget.currentWidget()
        if w is self.consoleTabPage:
            # FIXME: Scroll console log.
            #self.consoleTabPage.scroll()
            pass
        elif w is self.imagesTabPage:
            # Scroll in & activate search results.
            view = self.imagesTableView
            model = view.model()
            selectionModel = view.selectionModel()
            index = selectionModel.currentIndex()
            nextIndex = None
            if not index.isValid():
                nextIndex = model.index(0, 0)
            else:
                nextIndex = index.siblingAtRow(index.row() + (-1 if pageUp else 1))
            if not nextIndex.isValid():
                return
            selectionModel.setCurrentIndex(nextIndex, QItemSelectionModel.SelectCurrent)
            self.searchResultsActivated(nextIndex)

    def loadModules(self):
        global query_index
        if query_index == None:
            self.appendSearchOutput("Loading query-index...")
            qApp.processEvents()
            query_index = __import__('query-index')  # TODO: Adjust file name.
            self.appendSearchOutput("Loaded query-index.")
            qApp.processEvents()
        if self.search == None:
            self.appendSearchOutput("Instantiating search...")
            qApp.processEvents()
            self.search = query_index.Search()
            self.appendSearchOutput("Instantiated search.")
            qApp.processEvents()

            self.appendSearchOutput("\n" + self.search.init_msg)
            self.searchHint.setText("Short help: " + self.search.prompt_prefix)

    def showEvent(self, ev):
        super(MainWindow, self).showEvent(ev)
        QTimer.singleShot(0, self.delayLoadModules)  # (Run after all events.)

    def delayLoadModules(self):
        QTimer.singleShot(50, self.loadModules)  # (Delay a bit further in the hopes it might actually work.)

    def appendSearchOutput(self, lines):
        # Skip calls with nothing to convey.
        if lines == None or lines == "":
            return
        # Strip last newline, but only that.
        # That way, an empty line at the end
        # can be requested by "...\n\n".
        # Otherwise, we could simply use: lines.rstrip('\n')
        if lines[-1] == '\n':
            lines = lines[:-1]
        self.searchOutput.append(lines)

    def stdoutSearchOutput(self, code):
        ret = None
        with contextlib.closing(StringIO()) as f:
            with contextlib.redirect_stdout(f):
                ret = code()
            self.appendSearchOutput(f.getvalue())
        return ret

    def handleSearchInput(self):
        inputText  = self.searchInput.currentText()
        inputIndex = self.searchInput.currentIndex()
        storedText = None if inputIndex == -1 else self.searchInput.itemText(inputIndex)

        search = self.search
        if search == None:
            self.appendSearchOutput("Search not ready, yet...")
            return
        self.appendSearchOutput(">>> " + inputText)
        search.in_text = inputText.strip()
        if storedText != inputText:
            self.searchInput.addItem(inputText)
        self.searchInput.clearEditText()
        self.clearSearchResultsModel()

        iteration_done = self.stdoutSearchOutput(search.do_command)
        if iteration_done:
            # Check for q (quit) command.
            if search.running_cli == False:
                self.close()
            return

        self.stdoutSearchOutput(search.do_search)
        n_results = 0 if search.results is None else len(search.results)
        if not n_results > 0:
            self.appendSearchOutput("No results.")
            return
        #self.stdoutSearchOutput(search.do_display)
        self.appendSearchOutput(f"Building results model for {n_results} results...")
        j = 0
        while j < n_results:
            result, j, _ = search.prepare_result(j)
            if j is None:
                break
            elif result is None:
                continue
            self.appendToSearchResultsModel(result)
        self.appendSearchOutput(f"Built results model with {self.searchResultsModel.rowCount()} entries.")

    def createSearchResultsModel(self):
        model = QStandardItemModel(0, 4)
        model.setHorizontalHeaderLabels(["score", "fix_idx", "face_id", "filename"])
        self.searchResultsModel = model
        self.imagesTableView.setModel(model)
        self.imagesTableView.horizontalHeader().setStretchLastSection(True)

    def clearSearchResultsModel(self):
        self.searchResultSelected = None
        #
        # Don't use clear(), as that will get rid of the header labels
        # and column count, too...
        #self.searchResultsModel.clear()
        self.searchResultsModel.setRowCount(0)
        #
        self.imageLabel.clear()

    def appendToSearchResultsModel(self, result):
        model = self.searchResultsModel
        scoreItem  = QStandardItem(str(result.score))
        fixIdxItem = QStandardItem(str(result.fix_idx))
        faceIdItem = QStandardItem(str(result.face_id))
        tfnItem    = QStandardItem(str(result.tfn))
        items = [scoreItem, fixIdxItem, faceIdItem, tfnItem]
        for item in items:
            item.setData(result)
        model.appendRow(items)

    def searchResultsActivated(self, index):
        result = index.data(Qt.UserRole + 1)
        self.showSearchResult(result)

    def showSearchResult(self, result):
        if self.searchResultSelected is result:
            return
        self.searchResultSelected = result
        if result is None:
            return
        self.appendSearchOutput(result.format_output())
        # Prepare image.
        try:
            size = self.imageLabel.maximumSize()
            max_res = (size.width(), size.height())
            image = self.search.prepare_image(result, max_res)
            if image is None:
                raise RuntimeError("No image.")
        except Exception as ex:
            self.appendSearchOutput(f"Error preparing image: {ex}")
            return
        # Convert prepared image to Qt/GUI.
        qtImage = QImage(image.data, image.shape[1], image.shape[0], 3 * image.shape[1], QImage.Format_RGB888).rgbSwapped()
        self.imageLabel.setPixmap(QPixmap.fromImage(qtImage))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("CLI-P GUI")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
